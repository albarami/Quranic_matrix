# QBM Implementation Plan: Backend to LEGENDARY

> **Non-Negotiable Goal**: Fix QBM once and for all so it runs end-to-end (backend first, then frontend) with full capability, no fake numbers, no hardcoding to specific benchmark questions, and no "fallback becomes main."

## ⚠️ HARD RULES (ENFORCED)

### Rule 1: No Frontend Until Backend is LEGENDARY
Phase 5 (frontend) is **BLOCKED** until Phase 4 achieves:
- ≥180/200 PASS on benchmark
- 0 generic default verses
- 0 structured-intent fallbacks

### Rule 2: Wrap Existing Assets — No Re-Implementation
All planners/routing/payload work **MUST** wrap and reuse existing truth-layer assets:
- `data/graph/semantic_graph_v2.json`
- `data/evidence/concept_index_v2.jsonl`
- `data/evidence/evidence_index_v2_chunked.jsonl`
- `src/ml/legendary_planner.py` (LegendaryPlanner class)
- `src/ml/qbm_bouzidani_taxonomy.py` (Bouzidani taxonomy)
- `vocab/canonical_entities.json`

**DO NOT** re-implement graph traversal, entity resolution, or evidence retrieval. Use what exists.

## Definition of Done

- [ ] **≥180/200 PASS** on `qbm_legendary_200.v1.jsonl` benchmark
- [ ] **0 generic default verses** (no Surah 1 or early Baqarah fallbacks)
- [ ] **0 fabricated stats** (all numbers computed from payload, never LLM-invented)
- [ ] **0 "fallback_used_for_structured_intent"** patterns
- [ ] All answers produced deterministically from computed payload (LLM optional rephrase only)
- [ ] All planners fail-closed when evidence is missing

---

## Phase 0 — Lock the Rules (No Fabrication, No Generic Defaults, Fail-Closed)

**Objective**: Prevent the system from returning misleading "nice answers" with wrong evidence.

### 0.1 Kill "generic opening verses default" everywhere

- [x] Search and remove any logic that inserts Surah 1 (or early Baqarah) verses as fallback when retrieval fails
- [x] Replace with fail-closed:
  - `status="no_evidence"`
  - `debug.fail_closed_reason=<reason>`
  - Empty evidence arrays

### 0.1b Unify canonical tafsir sources to 7 everywhere

- [x] Create `src/ml/tafsir_constants.py` with `CANONICAL_TAFSIR_SOURCES` (7 sources)
- [x] Update `src/ml/legendary_planner.py` to import from `tafsir_constants.py`
- [x] Update `src/ml/full_power_system.py` to import from `tafsir_constants.py`
- [x] Update `src/ml/mandatory_proof_system.py` to import from `tafsir_constants.py`
- [x] Add test: `tests/test_tafsir_source_count.py` (6/6 PASSED)

### 0.2 Tests (must pass)

- [x] Add `tests/test_no_generic_default_verses.py` (12/12 PASSED):
  - For a query known to fail retrieval, assert no verses are returned
  - For a query that should retrieve, assert verses are not dominated by `{1:1-7, 2:1-20}` unless the query explicitly asks for those
  - Scoring detects generic default verses and fails appropriately
  - Intent classifier routes analytical queries correctly (not FREE_TEXT)

### Acceptance Gate

```bash
python -m pytest tests/test_no_generic_default_verses.py -v
```

**All tests must pass.**

### Commit Message

```
fix(truth): remove generic default verses; enforce fail-closed no-evidence
```

### Test Output Path

> - `reports/test_runs/phase0_test_tafsir_source_count_20251229.txt`
> - `reports/test_runs/phase0_test_no_generic_default_verses_20251229.txt`

---

## Phase 1 — Unify Routing and Stop "Wrong Component for the Job"

**Objective**: The main failure pattern is routing analytical questions into FREE_TEXT + vector search, which returns junk. Analytical questions must route to deterministic planners (graph/taxonomy/metrics), not RAG.

### 1.1 Create canonical `question_class_router.py`

- [x] File: `src/ml/question_class_router.py`
- [x] Must produce a `QuestionClass` (NOT benchmark ID, NOT hardcoded):
  - `CAUSAL_CHAIN`
  - `CROSS_TAFSIR_COMPARATIVE`
  - `BEHAVIOR_PROFILE_11AXIS`
  - `NETWORK_CENTRALITY`
  - `STATE_TRANSITION`
  - `AGENT_ATTRIBUTION`
  - `TEMPORAL_MAPPING`
  - `SEMANTIC_LANDSCAPE`
  - `COMPLETE_ANALYSIS` (used for consequence + integration until specialized planners land)
  - Fallback: `FREE_TEXT` (only for genuinely open-ended questions)

**Rules**:
- No benchmark-ID matching. Only semantic patterns + canonical entity extraction.
- Must work for similar phrasing in Arabic/English.

### 1.2 Update `MandatoryProofSystem` to call right planner

- [x] Updated `_route_query()` to use canonical `question_class_router`
- [x] Returns `question_class` for planner routing

### 1.3 Tests (must pass)

- [x] `tests/test_router_classification.py` (27/27 PASSED): with 2 variants per class (Arabic + English paraphrase)
- [x] Assert `class != FREE_TEXT` for analytics

### Acceptance Gate

```bash
python -m pytest tests/test_router_classification.py -v
```

**All tests must pass.**

### Commit Message

```
feat(router): unify question-class routing for analytical queries (no FREE_TEXT for analytics)
```

### Test Output Path

> - `reports/test_runs/phase1_test_router_classification_20251229.txt`

---

## Phase 2 — Build Deterministic "Analysis Payload" and Answer Generation (No Hallucinated Numbers) ✅ COMPLETED

**Objective**: LLM must not "invent" counts/percentages. Backend must compute them and the answer must be generated from computed payload.

### 2.1 Build deterministic analysis payload

- [x] Create: `src/benchmarks/analysis_payload.py`
- [x] Implement `build_analysis_payload(question, question_class, proof, debug) -> AnalysisPayload`
- [x] Payload includes:
  - Extracted canonical entities (IDs + labels) via `EntityInfo` dataclass
  - Computed graph outputs (paths/cycles/metrics) via `GraphOutput` dataclass
  - Computed tables (counts, consensus %, rankings) via `ComputedTable` dataclass
  - Evidence bundles (verse_keys + chunk_id + char_start/end + source) via `EvidenceBundle` dataclass
  - Explicit "gaps" list when data is missing
  - `computed_numbers` dict for validator gate
  - `derivations` dict for audit trail

### 2.2 Deterministic answer generator (baseline)

- [x] Create: `src/benchmarks/answer_generator.py`
- [x] Implement: `generate_answer(payload) -> str`
  - Deterministic, professional Arabic output, structured headings
  - **Numbers ONLY from computed payload** — LLM cannot invent counts/percentages
  - Optional: LLM rewriter via `generate_answer_with_llm_rewrite()`
  - **MANDATORY validator gate**: `validate_no_new_claims(payload, llm_output)` that:
    - Extracts all numbers from LLM output
    - Verifies each number exists in `payload.get_all_numbers()`
    - Rejects if any new number/claim found
    - Returns `(is_valid, violations)` tuple

### 2.3 Benchmark scoring must evaluate answer depth + correctness

- [x] Update scoring: `src/benchmarks/scoring.py`
- [x] PASS requires:
  - Non-empty answer (not placeholder) - `_is_placeholder_answer()` check
  - Required sections exist based on `expected.must_include` - `_check_required_sections()`
  - Required computed outputs exist (e.g., `min_hops` chains for MULTIHOP capability)
  - Evidence is cited and matches payload references (PROVENANCE capability)
  - "Disallow" checks enforced - `_check_disallow_violations()`

### 2.4 Tests (must pass)

- [x] `tests/test_answer_generation_deterministic.py` - 26 tests
- [x] `tests/test_scoring_depth_rules.py` - 32 tests

### Acceptance Gate

```bash
python -m pytest tests/test_answer_generation_deterministic.py tests/test_scoring_depth_rules.py -v
# Result: 58 passed
```

**All tests passed.**

### Commit Message

```
feat(benchmark): add validator gate for LLM output + derivations tracking (Phase 2)
```

### Test Output Path

> - `reports/test_runs/phase2_test_answer_generation_20251229.txt`
> - `reports/test_runs/phase2_test_scoring_depth_20251229.txt`

---

## Phase 3 — Extend LegendaryPlanner (No Re-Implementation) ✅ COMPLETED

**Objective**: Wrap and extend `LegendaryPlanner` to cover all 10 benchmark question classes. **DO NOT** create new planners that re-implement logic already in `LegendaryPlanner`.

### Hard Rule: Wrap, Don't Re-Implement

`LegendaryPlanner` already has:
- `QuestionClass` enum (25 classes)
- `detect_question_class()` for routing
- `create_plan()` for execution plans
- `execute_plan()` for running plans
- `resolve_entities()` using canonical entities
- `get_concept_evidence()` from concept index
- `get_semantic_neighbors()` from semantic graph
- `find_causal_paths()` for graph traversal

**Strategy**: Add missing execution logic to `LegendaryPlanner.execute_plan()` for question classes that need it. Create thin wrapper modules in `src/ml/planners/` ONLY if they add new capability (not duplicate).

Each planner returns a structured object that becomes proof + analysis payload.

### 3.1 CAUSAL_CHAIN planner

- [x] File: `src/ml/planners/causal_chain_planner.py` (thin wrapper)
- [x] **Reuse**: `LegendaryPlanner.find_causal_paths()` + `resolve_entities()`
- [x] **Add**: Multi-hop chain formatting for `min_hops` requirement
- [x] **Add**: Edge-level provenance attachment from semantic graph edges
- [x] **Add**: Tafsir quote retrieval from `evidence_index_v2_chunked`

### 3.2 CROSS_TAFSIR_COMPARISON planner

- [x] File: `src/ml/planners/cross_tafsir_planner.py` (thin wrapper)
- [x] **Reuse**: `LegendaryPlanner.get_concept_evidence()` + `CORE_SOURCES` (now 7)
- [x] **Add**: Agreement/disagreement metric computation across 7 sources
- [x] **Add**: Verse enumeration from concept index for target concepts

### 3.3 PROFILE_11D planner

- [x] File: `src/ml/planners/profile_11d_planner.py` (thin wrapper)
- [x] **Reuse**: `LegendaryPlanner.resolve_entities()` + `get_concept_evidence()`
- [x] **Reuse**: `qbm_bouzidani_taxonomy.py` for 11-axis classification
- [x] **Add**: Gap labeling when axis data missing (fail-closed, no invention)

### 3.4 GRAPH_METRICS planner

- [x] File: `src/ml/planners/graph_metrics_planner.py` (thin wrapper)
- [x] **Reuse**: `LegendaryPlanner.semantic_graph` (already loaded)
- [x] **Add**: Centrality metrics computation (degree, in/out)
- [x] **Add**: Table formatting for payload output

### 3.5 HEART_STATE planner

- [x] File: `src/ml/planners/heart_state_planner.py` (thin wrapper)
- [x] **Reuse**: `LegendaryPlanner.canonical_entities` (heart_states section)
- [x] **Reuse**: `get_concept_evidence()` for each heart state
- [x] **Add**: Transition graph from semantic graph edges (evidence-backed only)

### 3.6 AGENT planner

- [x] File: `src/ml/planners/agent_planner.py` (thin wrapper)
- [x] **Reuse**: `LegendaryPlanner.canonical_entities` (agents section)
- [x] **Reuse**: `get_concept_evidence()` + `get_semantic_neighbors()`
- [x] **Add**: Agent→behavior mapping aggregation

### 3.7 TEMPORAL_SPATIAL planner

- [x] File: `src/ml/planners/temporal_spatial_planner.py` (thin wrapper)
- [x] **Reuse**: `vocab/temporal.json`, `vocab/spatial.json`
- [x] **Reuse**: `LegendaryPlanner.get_concept_evidence()`
- [x] **Add**: Mapping aggregation for payload output

### 3.8 CONSEQUENCE planner

- [x] File: `src/ml/planners/consequence_planner.py` (thin wrapper)
- [x] **Reuse**: `LegendaryPlanner.canonical_entities` (consequences section)
- [x] **Reuse**: `get_semantic_neighbors()` with edge_types=["RESULTS_IN", "LEADS_TO"]
- [x] **Add**: Behavior→consequence mapping from graph edges

### 3.9 EMBEDDINGS planner (optional, must be truthful)

- [x] File: `src/ml/planners/embeddings_planner.py` (thin wrapper)
- [x] **Reuse**: Existing embedding index if available
- [x] **Add**: Model limitation disclosure from `data/models/registry.json`
- [x] **Rule**: No accuracy claims without evaluation proof

### 3.10 INTEGRATION_E2E planner

- [x] File: `src/ml/planners/integration_planner.py` (orchestrator)
- [x] **Reuse**: All other planners
- [x] **Add**: Cross-component consistency checks
- [x] **Add**: Conflict flagging in debug trace

### 3.11 Tests (must pass)

- [x] `tests/test_planners_smoke.py` (24 tests - fast smoke with mocks)
- [x] `tests/test_no_fabrication_all_planners.py` (29 tests - fail-closed, no fabrication; mocks)
- [x] `tests/test_planners_integration_real.py` (22 tests - REAL data, NO MOCKS)
  - Tests against real `vocab/canonical_entities.json` (73 behaviors, 14 agents, 16 consequences, 12 heart states)
  - Tests against real `vocab/temporal.json`, `vocab/spatial.json`
  - Tests against real `data/models/registry.json`
  - Tests against real `data/graph/semantic_graph_v2.json`
  - Validates 7 canonical tafsir sources
  - Validates 11 Bouzidani dimensions

### Acceptance Gate

```bash
python -m pytest tests/test_planners_smoke.py tests/test_no_fabrication_all_planners.py tests/test_planners_integration_real.py -v
# Result: 75 passed (24 smoke + 29 no-fabrication + 22 integration real)
```

**All tests passed (including REAL-data integration).**

### Commit Message

```
feat(planners): implement all 10 thin wrapper planners (Phase 3)
```

### Test Output Path

> - `reports/test_runs/phase3_test_planners_smoke_20251229.txt`
> - `reports/test_runs/phase3_test_no_fabrication_all_planners_20251229.txt`
> - `reports/test_runs/phase3_test_planners_integration_real_20251229.txt`

---

## Phase 4 — GLOBAL/CORPUS-WIDE Entity-Free Planners + Benchmark Loop

**CRITICAL CLARIFICATION**: "No explicit entity in the question" is a **SUPPORTED query type**, not a failure. Entity-free analytical questions (graph-wide, corpus-wide) MUST work.

### 4.0 Entity-Free Query Support (MANDATORY)

**Problem**: Questions like "Find all cycles A→B→C→A" or "Causal density analysis" don't contain specific Arabic behavior terms, but they are valid analytical queries that operate over the entire graph/corpus.

**Solution**: Extend `LegendaryPlanner.execute_plan()` to support entity-free analytical questions by operating over:
- `data/graph/semantic_graph_v2.json` (global graph analytics)
- `data/evidence/concept_index_v2.jsonl` (global concept frequency/coverage)
- `data/evidence/evidence_index_v2_chunked.jsonl` (tafsir evidence aggregation)
- `vocab/canonical_entities.json` (complete enumerations)

**Examples that MUST work without entities**:
- "Causal density" → compute top outgoing/incoming CAUSES over entire graph
- "Find all cycles A→B→C→A" → cycle detection over entire graph
- "Causal chain length distribution" → global path-length stats
- "Top 5 behaviors per tafsir source" → global counts from concept index
- "Complete consequence inventory" → enumerate canonical consequences + evidence

**Routing Rule Update**:
- If `question_class` is analytic (graph metrics, cycles, ranking, inventories), route to planner even when `resolved_entities == []`
- `FREE_TEXT` is ONLY for genuinely open-ended text questions, NEVER for benchmark-style analytics

**Fail-Closed Clarification**:
- Fail-closed only when the required substrate truly cannot support the request
- For graph-wide queries, evidence IS the graph + its edge provenance; do NOT return empty
- "No entity" must NOT imply "no evidence"

**Correctness Gates**:
- If entities were resolved: returned evidence must intersect those entities' evidence sets
- If entity-free planner: Quran verses are optional; prefer computed tables + cited provenance edges

### 4.1 Benchmark Commands

**Smoke (20)**:
```bash
python scripts/run_qbm_benchmark.py --dataset data/benchmarks/qbm_legendary_200.v1.jsonl --smoke --ci
```

**Full (200)**:
```bash
python scripts/run_qbm_benchmark.py --dataset data/benchmarks/qbm_legendary_200.v1.jsonl
```

### 4.2 Remediation Loop

For each FAIL:
1. Identify which planner/class failed
2. Fix planner capability (NOT by question ID)
3. Add a regression test capturing the failure pattern class-level
4. Re-run smoke until 20/20 PASS
5. Then re-run full

### 4.3 Acceptance Gate

- [ ] Smoke: **20/20 PASS**
- [ ] Full: **≥180/200 PASS**
- [ ] **0** "generic opening verses default"
- [ ] **0** "fallback_used_for_structured_intent"
- [ ] **0** fabricated numbers (validator catches any number not in payload)

### Commit Messages (per remediation batch)

```
fix(benchmark): improve <planner> for <question_class>; add regression tests
```

### Benchmark Output Paths

> _To be filled after execution_

---

## Phase 5 — Frontend Integration + E2E

> ⛔ **BLOCKED**: This phase cannot begin until Phase 4 gate passes (≥180/200 PASS, 0 generic defaults, 0 structured-intent fallbacks)

**Objective**: UI must display backend truths; no markdown-as-text; no LLM-generated stats.

### 5.1 UI must never compute or invent metrics

- [ ] Frontend charts/tables must be rendered from:
  - `/api/metrics/overview`
  - `/api/proof/query` proof payload
  - `/api/genome/*`
  - `/api/reviews/*`

**Chat UI rules**:
- If user asks for stats/plots, frontend calls metrics endpoint and renders components
- LLM may only provide narrative explanation referencing returned JSON

### 5.2 Playwright E2E (blocking)

- [ ] Must pass on CI (remove `continue-on-error` once stable)
- [ ] Tests:
  - Proof query render (Arabic headers, tables, citations)
  - Pagination working
  - Genome export download
  - Reviews workflow

### Acceptance Gate

```bash
cd qbm-frontendv3 && npx playwright test
```

**All E2E tests must pass.**

### Commit Message

```
feat(frontend): render proof/metrics with styled components + E2E gates
```

### Test Output Path

> _To be filled after execution_

---

## Repo Hygiene and Use-Everything Policy

We use everything that adds value without compromising truth:

| Module | Integration | Status |
|--------|-------------|--------|
| `qbm_bouzidani_taxonomy.py` | PROFILE_11D planner | **Mandatory** |
| `cross_context_behavior_handler` | Cross-context class | **Mandatory if present** |
| Domain reranker | Retrieval improvement | Optional, measured |
| PyG/GNN | Graph analysis | Opt-in, never required for correctness |

---

## Mandatory Workflow Rules (Every Phase)

1. Update this file (`docs/IMPLEMENTATION_PLAN_BACKEND_TO_LEGENDARY.md`) with:
   - What changed
   - Exact test commands executed
   - Benchmark output paths

2. Run tests required for that phase

3. Commit and push:
   ```bash
   git add . && git commit -m "<message>" && git push origin main
   ```

4. **No skipping.**

---

## Execution Log

### Phase 0 Execution

**Date**: 2025-12-29

**Changes Made**:
- Created `src/ml/tafsir_constants.py` with canonical 7-source tafsir list
- Updated `src/ml/legendary_planner.py` to import from `tafsir_constants.py`
- Updated `src/ml/full_power_system.py` to import from `tafsir_constants.py` (6 locations)
- Updated `src/ml/mandatory_proof_system.py` to import from `tafsir_constants.py`
- Added `debug.fail_closed_reason` to proof debug schema and API payloads (prevents crash on no-evidence)
- Added top-level `status="no_evidence"` when both Quran + tafsir evidence are empty
- Updated remaining `src/ml/*` modules to import tafsir sources from `tafsir_constants.py` (no hardcoded lists)
- Updated `src/api/routers/proof.py` to use `CANONICAL_TAFSIR_SOURCES` (no hardcoded lists)
- Created `tests/test_tafsir_source_count.py` (6 tests)
- Created `tests/test_no_generic_default_verses.py` (12 tests)
- Updated tests to assert `fail_closed_reason` field exists

**Test Commands**:
```bash
python -m pytest tests/test_tafsir_source_count.py -v  # 6/6 PASSED
python -m pytest tests/test_no_generic_default_verses.py -v  # 12/12 PASSED
```

**Test Output**: All 18 tests PASSED

**Commit Hash**: f3179098700b77fe4d9f5059c150636c2eb58015

---

### Phase 1 Execution

**Date**: 2025-12-29

**Changes Made**:
- Created `src/ml/question_class_router.py` - canonical router wrapping intent_classifier + legendary_planner
- Updated `src/ml/mandatory_proof_system.py` `_route_query()` to use canonical router
- Created `tests/test_router_classification.py` (27 tests)
- All 45 Phase 0+1 tests passing

**Test Command**:
```bash
python -m pytest tests/test_router_classification.py -v  # 27/27 PASSED
```

**Test Output**: All 27 tests PASSED

**Commit Hash**: f3179098700b77fe4d9f5059c150636c2eb58015

---

### Phase 2 Execution

**Date**: 2025-12-29

**Changes Made**:
- Added deterministic Phase 2 payload + answer generation modules (`src/benchmarks/analysis_payload.py`, `src/benchmarks/answer_generator.py`)
- Enhanced benchmark scoring depth/correctness rules (`src/benchmarks/scoring.py`)
- Added Phase 2 test suites (58 tests total)
- Integrated deterministic payload + validator-gated LLM rewrite into `src/ml/mandatory_proof_system.py` (fail-closed on invented numbers)

**Test Command**:
```bash
python -m pytest tests/test_answer_generation_deterministic.py tests/test_scoring_depth_rules.py -v
```

**Test Output**:
> - `reports/test_runs/phase2_test_answer_generation_20251229.txt`
> - `reports/test_runs/phase2_test_scoring_depth_20251229.txt`
> - Result: 58 passed

**Commit Hash**: f3179098700b77fe4d9f5059c150636c2eb58015

---

### Phase 3 Execution

**Date**: 2025-12-29

**Changes Made**:
- Created 10 Phase 3 thin-wrapper planners under `src/ml/planners/` (no re-implementation of `LegendaryPlanner`)
- Added orchestration + cross-component checks in `src/ml/planners/integration_planner.py`
- Added planner validation test suites:
  - Smoke (24) + no-fabrication (29) using mocks
  - Integration (22) against REAL files (no mocks)

**Test Command**:
```bash
python -m pytest tests/test_planners_smoke.py tests/test_no_fabrication_all_planners.py tests/test_planners_integration_real.py -v
```

**Test Output**:
> - `reports/test_runs/phase3_test_planners_smoke_20251229.txt`
> - `reports/test_runs/phase3_test_no_fabrication_all_planners_20251229.txt`
> - `reports/test_runs/phase3_test_planners_integration_real_20251229.txt`
> - Result: 75 passed

**Commit Hash**: f3179098700b77fe4d9f5059c150636c2eb58015

---

### Phase 4 Execution

**Date**: _To be filled_

**Smoke Benchmark Command**:
```bash
python scripts/run_qbm_benchmark.py --dataset data/benchmarks/qbm_legendary_200.v1.jsonl --smoke --ci
```

**Smoke Result**: _/_ PASS

**Full Benchmark Command**:
```bash
python scripts/run_qbm_benchmark.py --dataset data/benchmarks/qbm_legendary_200.v1.jsonl
```

**Full Result**: _/200 PASS

**Benchmark Report Path**: _To be filled_

**Commit Hash**: _To be filled_

---

### Phase 5 Execution

**Date**: _To be filled_

**Changes Made**:
> _To be filled_

**E2E Test Command**:
```bash
cd qbm-frontendv3 && npx playwright test
```

**Test Output**:
> _To be filled_

**Commit Hash**: _To be filled_

---

## Final Sign-Off

- [ ] ≥180/200 PASS on benchmark
- [ ] 0 generic default verses
- [ ] 0 fabricated stats
- [ ] 0 fallback_used_for_structured_intent
- [ ] All planners fail-closed when evidence missing
- [ ] Frontend renders backend truths only
- [ ] E2E tests passing

**Signed off by**: _________________

**Date**: _________________

**Final Commit Hash**: _________________
