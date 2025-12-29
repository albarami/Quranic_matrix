# QBM Codebase Audit Report
**Date:** 2025-12-28  
**Purpose:** Comprehensive assessment before benchmark fixes

---

## EXECUTIVE SUMMARY

The QBM codebase has **extensive existing infrastructure** that is **NOT being fully utilized**. The benchmark is failing because:

1. **LegendaryPlanner exists but is only partially integrated** into the full power path
2. **Multiple routing systems exist** causing confusion (intent_classifier, query_router, legendary_planner's detect_question_class)
3. **Generic fallback verses were being inserted** when retrieval failed (NOW REMOVED)
4. **Taxonomy modules exist but are NOT used** for 11-axis profiles

---

## SECTION 1: EXISTING ML MODULES (46 files in src/ml/)

### 1.1 PLANNERS & ROUTERS (CRITICAL - Should be primary path)

| Module | Lines | Purpose | Integration Status |
|--------|-------|---------|-------------------|
| `legendary_planner.py` | 690 | **25 question classes**, entity resolution, concept evidence, graph traversal, causal paths | ⚠️ PARTIAL - only used in proof_only_backend |
| `query_planner.py` | 582 | Query planning with debug trace | ❌ NOT USED |
| `query_router.py` | 462 | Intent classification (AYAH_REF, SURAH_REF, CONCEPT_REF, FREE_TEXT) | ⚠️ PARTIAL |
| `intent_classifier.py` | 385 | Benchmark intent classification (10 intents) | ⚠️ PARTIAL - added recently |
| `routed_evidence_retriever.py` | 306 | Query-routed deterministic + hybrid | ❌ NOT USED |

### 1.2 EVIDENCE RETRIEVAL (CRITICAL)

| Module | Lines | Purpose | Integration Status |
|--------|-------|---------|-------------------|
| `hybrid_evidence_retriever.py` | 818 | BM25 + Dense retrieval over chunked tafsir | ✅ USED |
| `stratified_retriever.py` | 441 | Guaranteed 7-source tafsir coverage | ⚠️ PARTIAL |
| `cross_context_behavior_handler.py` | 757 | Cross-context behavioral analysis | ❌ NOT USED |

### 1.3 FULL POWER SYSTEM (CRITICAL)

| Module | Lines | Purpose | Integration Status |
|--------|-------|---------|-------------------|
| `full_power_system.py` | 1242 | GPU RAG, embeddings, reranker, LLM | ✅ USED |
| `mandatory_proof_system.py` | 1447 | 15-component proof generation | ✅ USED - but has issues |
| `proof_only_backend.py` | 600+ | Lightweight deterministic backend | ✅ USED |

### 1.4 GRAPH & REASONING

| Module | Lines | Purpose | Integration Status |
|--------|-------|---------|-------------------|
| `graph_reasoner.py` | 787 | GNN multi-hop reasoning | ❌ NOT USED (PyG dependency) |
| `relation_extractor.py` | - | Behavior relation extraction | ❌ NOT USED |

### 1.5 TAXONOMY MODULES (NOT USED - Should be for 11-axis)

| Module | Purpose | Integration Status |
|--------|---------|-------------------|
| `qbm_5axis_schema.py` | Bouzidani 5-axis classification | ❌ NOT USED |
| `qbm_bouzidani_taxonomy.py` | Full Bouzidani taxonomy | ❌ NOT USED |
| `qbm_usul_taxonomy.py` | Usul-based taxonomy | ❌ NOT USED |
| `quranic_behavior_taxonomy.py` | Quranic behavior taxonomy | ❌ NOT USED |

### 1.6 EMBEDDINGS & CLASSIFICATION

| Module | Purpose | Integration Status |
|--------|---------|-------------------|
| `arabic_embeddings.py` | Arabic embedding pipeline | ✅ USED |
| `behavioral_classifier.py` | Behavior classification | ❌ NOT USED |
| `domain_reranker.py` | Domain-specific reranking | ⚠️ PARTIAL |
| `embedding_pipeline.py` | Embedding generation | ✅ USED |

---

## SECTION 2: DATA FILES

### 2.1 EVIDENCE INDEXES (data/evidence/)

| File | Size | Version | Used By |
|------|------|---------|---------|
| `concept_index_v2.jsonl` | 7.1 MB | v2 ✅ | legendary_planner, mandatory_proof_system, proof_only_backend |
| `evidence_index_v2_chunked.jsonl` | 119 MB | v2 ✅ | hybrid_evidence_retriever |
| `concept_index_v1.jsonl` | 3.4 MB | v1 ⚠️ | query_planner (OUTDATED) |
| `evidence_index_v1.jsonl` | 102 MB | v1 ⚠️ | (legacy) |

### 2.2 GRAPHS (data/graph/)

| File | Size | Version | Used By |
|------|------|---------|---------|
| `semantic_graph_v2.json` | 13.9 MB | v2 ✅ | legendary_planner |
| `cooccurrence_graph_v1.json` | 1.4 MB | v1 | legendary_planner |
| `semantic_graph_v1.json` | 3.3 MB | v1 ⚠️ | query_planner (OUTDATED) |

### 2.3 VECTOR INDEXES (data/indexes/)

| File | Size | Purpose |
|------|------|---------|
| `full_power_index.npy` | 362 MB | FAISS vector index |
| `full_power_index.npy.metadata.json` | 44 MB | Index metadata |

### 2.4 MODELS (data/models/)

| Directory | Purpose |
|-----------|---------|
| `qbm-arabic-embeddings/` | Arabic embedding model |
| `qbm-arabic-finetuned/` | Fine-tuned model |
| `qbm-behavioral-classifier/` | Behavior classifier |
| `qbm-domain-reranker/` | Domain reranker |
| `qbm-gnn-v2/` | GNN model |
| `qbm-graph-reasoner/` | Graph reasoner model |

---

## SECTION 3: VOCABULARY FILES (vocab/)

### 3.1 CORE ENTITY FILES

| File | Content |
|------|---------|
| `canonical_entities.json` | 73 behaviors, 14 agents, 26 organs, 12 heart states, 16 consequences |
| `entity_types.json` | Entity type mappings |
| `tafsir_sources.json` | 7 tafsir source definitions |

### 3.2 TAXONOMY FILES (11-Axis)

| File | Purpose |
|------|---------|
| `axes.json` | 11-axis definitions |
| `organs.json` | Organ classifications |
| `agents.json` | Agent types |
| `temporal.json` | Temporal classifications |
| `spatial.json` | Spatial classifications |
| `systemic.json` | Systemic classifications |
| `evaluation.json` | Moral evaluation |
| `polarity.json` | Behavior polarity |

---

## SECTION 4: LEGENDARY PLANNER CAPABILITIES

The `legendary_planner.py` already implements **25 question classes**:

### Category 1: Causal Chain Analysis
- `CAUSAL_CHAIN` - Q1: Path from A to B with evidence
- `SHORTEST_PATH` - Q2: Shortest transformation path
- `REINFORCEMENT_LOOP` - Q3: Feedback loops

### Category 2: Cross-Tafsir Comparative
- `CROSS_TAFSIR_COMPARATIVE` - Q4: Multi-source comparison
- `MAKKI_MADANI_ANALYSIS` - Q5: Revelation phase analysis
- `CONSENSUS_DISPUTE` - Q6: Agreement/disagreement

### Category 3: 11-Axis + Taxonomy
- `BEHAVIOR_PROFILE_11AXIS` - Q7: Full 11-axis profile
- `ORGAN_BEHAVIOR_MAPPING` - Q8: Organ-behavior mapping
- `STATE_TRANSITION` - Q9: Heart state transitions

### Category 4: Agent-Based Analysis
- `AGENT_ATTRIBUTION` - Q10: Agent attribution
- `AGENT_CONTRAST_MATRIX` - Q11: Agent comparison
- `PROPHETIC_ARCHETYPE` - Q12: Prophetic patterns

### Category 5: Network + Graph Analytics
- `NETWORK_CENTRALITY` - Q13: Degree/betweenness centrality
- `COMMUNITY_DETECTION` - Q14: Behavior clusters
- `BRIDGE_BEHAVIORS` - Q15: Bridge behaviors

### Category 6: Temporal + Spatial Context
- `TEMPORAL_MAPPING` - Q16: Temporal patterns
- `SPATIAL_MAPPING` - Q17: Spatial patterns

### Category 7: Statistics + Patterns
- `SURAH_FINGERPRINTS` - Q18: Surah behavior profiles
- `FREQUENCY_CENTRALITY` - Q19: Frequency analysis
- `MAKKI_MADANI_SHIFT` - Q20: Phase shifts

### Category 8: Embeddings + Semantics
- `SEMANTIC_LANDSCAPE` - Q21: Embedding space
- `MEANING_DRIFT` - Q22: Semantic drift

### Category 9: Complex Multi-System
- `COMPLETE_ANALYSIS` - Q23: Full analysis
- `PRESCRIPTION_GENERATOR` - Q24: Prescription generation
- `GENOME_ARTIFACT` - Q25: Complete export

---

## SECTION 5: IDENTIFIED ISSUES

### Issue 1: Multiple Routing Systems
**Problem:** Three different routing systems exist:
1. `intent_classifier.py` - 10 benchmark intents
2. `query_router.py` - 4 standard intents
3. `legendary_planner.detect_question_class()` - 25 question classes

**Impact:** Confusion about which router to use, inconsistent routing.

### Issue 2: LegendaryPlanner Not Fully Integrated
**Problem:** `mandatory_proof_system.py` only partially uses `LegendaryPlanner`.
- It calls `LegendaryPlanner.query()` but doesn't fully utilize the results
- It still does its own routing via `_route_query()`

**Impact:** Benchmark questions not routed to proper planners.

### Issue 3: Generic Fallback Verses (FIXED)
**Problem:** When retrieval failed, system inserted Surah 1 verses as fallback.

**Status:** ✅ FIXED - Removed fallback code, now fail-closed.

### Issue 4: Taxonomy Modules Not Used
**Problem:** For PROFILE_11D questions, we have taxonomy modules but they're not used.

**Impact:** 11-axis profiles not properly generated.

### Issue 5: query_planner.py Uses v1 Files
**Problem:** `query_planner.py` uses `concept_index_v1.jsonl` and `semantic_graph_v1.json`.

**Impact:** Outdated data if this planner is used.

---

## SECTION 6: RECOMMENDED FIXES

### Fix 1: Unify Routing
Use `LegendaryPlanner.detect_question_class()` as the PRIMARY router since it has the most comprehensive 25-class system.

### Fix 2: Full LegendaryPlanner Integration
In `mandatory_proof_system.answer_with_full_proof()`:
1. Call `LegendaryPlanner.query()` for ALL analytical questions
2. Use the planner's evidence directly
3. Don't duplicate routing logic

### Fix 3: Use Taxonomy Modules
For BEHAVIOR_PROFILE_11AXIS questions, integrate:
- `qbm_bouzidani_taxonomy.py`
- `qbm_5axis_schema.py`
- `vocab/axes.json`

### Fix 4: Update query_planner.py
Change from v1 to v2 files:
- `concept_index_v1.jsonl` → `concept_index_v2.jsonl`
- `semantic_graph_v1.json` → `semantic_graph_v2.json`

### Fix 5: Use cross_context_behavior_handler.py
For CROSS_CONTEXT_BEHAVIOR intent, route to this dedicated handler.

---

## SECTION 7: COMPONENT UTILIZATION MATRIX

| Benchmark Intent | Should Use | Currently Using |
|-----------------|------------|-----------------|
| GRAPH_CAUSAL | LegendaryPlanner.find_causal_paths() | ⚠️ Partial |
| CROSS_TAFSIR_ANALYSIS | LegendaryPlanner + hybrid_evidence_retriever | ⚠️ Partial |
| PROFILE_11D | LegendaryPlanner + taxonomy modules | ❌ Missing taxonomy |
| GRAPH_METRICS | LegendaryPlanner + networkx | ⚠️ Partial |
| HEART_STATE | LegendaryPlanner + canonical_entities | ⚠️ Partial |
| AGENT_ANALYSIS | LegendaryPlanner + canonical_entities | ⚠️ Partial |
| TEMPORAL_SPATIAL | LegendaryPlanner | ⚠️ Partial |
| CONSEQUENCE_ANALYSIS | LegendaryPlanner + canonical_entities | ⚠️ Partial |
| EMBEDDINGS_ANALYSIS | LegendaryPlanner + embedding_pipeline | ❌ Not integrated |
| INTEGRATION_E2E | All components | ⚠️ Partial |

---

## SECTION 8: ACTION PLAN

### Phase 1: Immediate (Before Next Benchmark Run)
1. ✅ Remove generic fallback verses (DONE)
2. ⬜ Ensure LegendaryPlanner.query() is called for all benchmark intents
3. ⬜ Use planner results directly for verse_keys and evidence

### Phase 2: Short-term
1. ⬜ Integrate taxonomy modules for 11-axis profiles
2. ⬜ Integrate cross_context_behavior_handler
3. ⬜ Update query_planner.py to use v2 files

### Phase 3: Medium-term
1. ⬜ Unify routing into single system
2. ⬜ Add graph_reasoner for GNN-based reasoning (if PyG available)
3. ⬜ Add answer quality scoring to benchmark

---

## APPENDIX: FILE COUNTS

| Directory | Python Files | JSON Files | Other |
|-----------|-------------|------------|-------|
| src/ml/ | 46 | - | - |
| src/api/ | 12 | - | - |
| scripts/ | 29 | - | - |
| vocab/ | - | 34 | - |
| data/evidence/ | - | 4 | 4 |
| data/graph/ | - | 3 | - |
| tests/ | 43 | - | - |

**Total:** 130+ Python files, 41+ JSON files

---

*Report generated: 2025-12-28*
