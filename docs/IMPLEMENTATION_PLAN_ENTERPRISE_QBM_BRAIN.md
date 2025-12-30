# Enterprise QBM Brain Implementation Plan

> **Version**: 1.0  
> **Created**: 2025-12-30  
> **Status**: Phase 0 - Planning  
> **Goal**: Zero-hallucination, Arabic-first, academically defensible system for the 200-question benchmark

---

## System Goals

### Core Principles

1. **Tool-first + Evidence-first**: The LLM is NEVER the source of truth. Postgres SSOT + graph + indices are truth.
2. **Every claim backed by verse_key and/or verse-keyed tafsir source**: No unsupported assertions.
3. **No "tafsir text contains term" relevance filtering**: Relevance is behavior→verse evidence. Tafsir fetched by exact verse_key × source.
4. **No silent fallbacks**: Missing data surfaces explicitly and fails tests.
5. **Arabic-first with Arabic+English mirror**: All canonical labels in Arabic, English translations provided.
6. **Academic template**: Every response must be reproducible and audit-trailed.

### Success Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Benchmark Pass Rate | ≥180/200 | Full 200-question suite |
| Generic Default Verses | 0 | No Surah 1/early Baqarah fallbacks |
| Fabricated Stats | 0 | All numbers from computed payload |
| Provenance Coverage | 100% | Every claim has verse_key + source |
| SSOT Integrity | 100% | No orphan records, all FKs valid |
| Behavior Inventory | 87 locked | Reconciled and frozen in Phase 1 |

---

## SSOT Schema (Postgres Tables + Constraints)

### Existing Schema (schemas/postgres_truth_layer.sql)

The following tables already exist. **SSOT status is defined in the SSOT Decision Appendix.**

| Table | Purpose | SSOT Status |
|-------|---------|-------------|
| `verses` | 6236 Quran verses with text_uthmani, text_simple | **SSOT** |
| `tafsir_chunks` | Chunked tafsir with verse_key × source | **SSOT** (extend with entry_type) |
| `entities` | Canonical entities (behaviors, agents, organs, etc.) | **SSOT** |
| `mentions` | Entity mentions in tafsir with provenance | **SSOT** |
| `semantic_edges` | Typed relationships between entities | **SSOT** |
| `edge_evidence` | Evidence supporting each edge | **SSOT** |

**Decision**: We extend existing tables rather than create duplicates. See SSOT Decision Appendix for full mapping.

### Required Extensions (Phase 1)

```sql
-- Extensions to existing tables + new tables
-- NOTE: We extend existing SSOT tables, not create duplicates

-- 1. Add evidence policy columns to entities table (the SSOT)
-- Store as proper columns, not in metadata JSONB (faster, validated)
ALTER TABLE entities ADD COLUMN IF NOT EXISTS evidence_mode VARCHAR(20) 
    CHECK (evidence_mode IN ('lexical', 'annotation', 'hybrid')) DEFAULT 'hybrid';
ALTER TABLE entities ADD COLUMN IF NOT EXISTS lexical_required BOOLEAN DEFAULT false;
ALTER TABLE entities ADD COLUMN IF NOT EXISTS allowed_evidence_types TEXT[] 
    DEFAULT ARRAY['EXPLICIT', 'IMPLICIT', 'CONTEXTUAL', 'TAFSIR_DERIVED'];
ALTER TABLE entities ADD COLUMN IF NOT EXISTS roots TEXT[];  -- Arabic roots as proper array
ALTER TABLE entities ADD COLUMN IF NOT EXISTS synonyms TEXT[];  -- Synonyms as proper array

-- Materialized view for convenience queries (reads from columns, not metadata)
CREATE MATERIALIZED VIEW behaviors AS
SELECT 
    entity_id as behavior_id,
    label_ar,
    label_en,
    category,
    polarity,
    roots,
    synonyms,
    evidence_mode,
    lexical_required,
    allowed_evidence_types
FROM entities
WHERE entity_type = 'BEHAVIOR';

CREATE UNIQUE INDEX idx_behaviors_id ON behaviors(behavior_id);

-- 2. behavior_verse_links (explicit behavior→verse evidence)
-- NOTE: FK references entities(entity_id), NOT the materialized view
-- Postgres cannot enforce FK against materialized views
CREATE TABLE behavior_verse_links (
    id SERIAL PRIMARY KEY,
    behavior_id VARCHAR(50) NOT NULL,
    surah INTEGER NOT NULL,
    ayah INTEGER NOT NULL,
    verse_key VARCHAR(10) GENERATED ALWAYS AS (surah || ':' || ayah) STORED,
    evidence_type VARCHAR(30) CHECK (evidence_type IN (
        'EXPLICIT', 'IMPLICIT', 'CONTEXTUAL', 'TAFSIR_DERIVED'
    )),
    directness VARCHAR(20) CHECK (directness IN ('direct', 'indirect', 'inferred')),
    provenance VARCHAR(100),
    confidence REAL DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- FK to entities table (SSOT), not materialized view
    FOREIGN KEY (behavior_id) REFERENCES entities(entity_id) ON DELETE CASCADE,
    FOREIGN KEY (surah, ayah) REFERENCES verses(surah, ayah) ON DELETE CASCADE,
    UNIQUE(behavior_id, surah, ayah, evidence_type)
);

-- Trigger to enforce behavior_id references only BEHAVIOR entities
CREATE OR REPLACE FUNCTION check_behavior_entity_type()
RETURNS TRIGGER AS $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM entities 
        WHERE entity_id = NEW.behavior_id AND entity_type = 'BEHAVIOR'
    ) THEN
        RAISE EXCEPTION 'behavior_id must reference an entity with entity_type=BEHAVIOR';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER enforce_behavior_type
    BEFORE INSERT OR UPDATE ON behavior_verse_links
    FOR EACH ROW EXECUTE FUNCTION check_behavior_entity_type();

-- 3. Extend tafsir_chunks (SSOT) with entry_type - DO NOT create tafsir_entries duplicate
ALTER TABLE tafsir_chunks ADD COLUMN IF NOT EXISTS entry_type VARCHAR(20) 
    CHECK (entry_type IN ('verse', 'surah_intro', 'appendix')) DEFAULT 'verse';
ALTER TABLE tafsir_chunks ADD COLUMN IF NOT EXISTS text_normalized TEXT;

-- Create index for entry_type filtering
CREATE INDEX IF NOT EXISTS idx_chunks_entry_type ON tafsir_chunks(entry_type);

-- 4. lexemes table (roots/lemmas/forms)
CREATE TABLE lexemes (
    id SERIAL PRIMARY KEY,
    root VARCHAR(20) NOT NULL,
    lemma VARCHAR(50),
    form VARCHAR(50),
    pos VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(root, lemma, form)
);

-- 5. verse_lexemes (verse→lexeme mapping)
CREATE TABLE verse_lexemes (
    id SERIAL PRIMARY KEY,
    surah INTEGER NOT NULL,
    ayah INTEGER NOT NULL,
    verse_key VARCHAR(10) GENERATED ALWAYS AS (surah || ':' || ayah) STORED,
    lexeme_id INTEGER NOT NULL,
    word_position INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (surah, ayah) REFERENCES verses(surah, ayah),
    FOREIGN KEY (lexeme_id) REFERENCES lexemes(id)
);

-- 6. Extend semantic_edges (SSOT) - DO NOT create graph_edges duplicate
-- semantic_edges is the SSOT for all typed relationships
ALTER TABLE semantic_edges ADD COLUMN IF NOT EXISTS from_entity_type VARCHAR(30);
ALTER TABLE semantic_edges ADD COLUMN IF NOT EXISTS to_entity_type VARCHAR(30);

-- provenance column = HIGH-LEVEL SUMMARY (counts, sources list)
-- edge_evidence table = DETAILED EVIDENCE ROWS (verse_keys, tafsir refs, spans)
-- Both are SSOT but serve different purposes
ALTER TABLE semantic_edges ADD COLUMN IF NOT EXISTS provenance JSONB;
COMMENT ON COLUMN semantic_edges.provenance IS 
    'High-level provenance summary: {sources_list, evidence_count, confidence}. Detailed evidence in edge_evidence table.';

-- Update entity types from entities table
UPDATE semantic_edges se SET 
    from_entity_type = (SELECT entity_type FROM entities WHERE entity_id = se.from_entity_id),
    to_entity_type = (SELECT entity_type FROM entities WHERE entity_id = se.to_entity_id)
WHERE from_entity_type IS NULL OR to_entity_type IS NULL;

-- 7. pgvector extension for embeddings (required for Section I benchmark)
CREATE EXTENSION IF NOT EXISTS vector;

-- 8. Embedding registry (model provenance for zero-hallucination)
CREATE TABLE IF NOT EXISTS embedding_registry (
    id SERIAL PRIMARY KEY,
    model_id VARCHAR(100) UNIQUE NOT NULL,
    model_name VARCHAR(200) NOT NULL,
    dimensions INTEGER NOT NULL,
    normalization VARCHAR(20) CHECK (normalization IN ('l2', 'cosine', 'none')) DEFAULT 'l2',
    training_date DATE,
    base_model VARCHAR(200),
    fine_tuned_on VARCHAR(200),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 9. Behavior embeddings table
CREATE TABLE IF NOT EXISTS behavior_embeddings (
    id SERIAL PRIMARY KEY,
    behavior_id VARCHAR(50) NOT NULL,
    model_id VARCHAR(100) NOT NULL,
    embedding vector(768),  -- AraBERT default; adjust per model
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (behavior_id) REFERENCES entities(entity_id) ON DELETE CASCADE,
    FOREIGN KEY (model_id) REFERENCES embedding_registry(model_id) ON DELETE CASCADE,
    UNIQUE(behavior_id, model_id)
);

-- 10. Verse embeddings table
CREATE TABLE IF NOT EXISTS verse_embeddings (
    id SERIAL PRIMARY KEY,
    surah INTEGER NOT NULL,
    ayah INTEGER NOT NULL,
    verse_key VARCHAR(10) GENERATED ALWAYS AS (surah || ':' || ayah) STORED,
    model_id VARCHAR(100) NOT NULL,
    embedding vector(768),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (surah, ayah) REFERENCES verses(surah, ayah) ON DELETE CASCADE,
    FOREIGN KEY (model_id) REFERENCES embedding_registry(model_id) ON DELETE CASCADE,
    UNIQUE(surah, ayah, model_id)
);

-- 11. Tafsir chunk embeddings table
CREATE TABLE IF NOT EXISTS tafsir_embeddings (
    id SERIAL PRIMARY KEY,
    chunk_id VARCHAR(100) NOT NULL,
    model_id VARCHAR(100) NOT NULL,
    embedding vector(768),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (chunk_id) REFERENCES tafsir_chunks(chunk_id) ON DELETE CASCADE,
    FOREIGN KEY (model_id) REFERENCES embedding_registry(model_id) ON DELETE CASCADE,
    UNIQUE(chunk_id, model_id)
);

-- Indexes for vector similarity search
CREATE INDEX IF NOT EXISTS idx_behavior_emb_vector ON behavior_embeddings USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_verse_emb_vector ON verse_embeddings USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_tafsir_emb_vector ON tafsir_embeddings USING ivfflat (embedding vector_cosine_ops);
```

### Arabic Normalization

```python
# Normalization PROFILES - not a single ruleset
# Evidence validation MUST use STRICT profile

NORMALIZATION_PROFILES = {
    'STRICT': {
        # Used for evidence validation - safe transforms only
        'remove_tashkeel': True,      # Remove diacritics
        'normalize_alef': True,       # أإآا → ا
        'normalize_yaa': True,        # ى → ي
        'remove_tatweel': True,       # Remove kashida ـ
        'normalize_taa_marbuta': False, # ة→ه is RISKY - disabled
        'normalize_hamza': False,      # Hamza transforms are RISKY - disabled
    },
    'LOOSE': {
        # Used ONLY for fallback discovery, NEVER for evidence validation
        'remove_tashkeel': True,
        'normalize_alef': True,
        'normalize_yaa': True,
        'remove_tatweel': True,
        'normalize_taa_marbuta': True,  # ة → ه (fallback only)
        'normalize_hamza': True,        # Hamza normalization (fallback only)
    }
}

# RULE: Evidence validation uses STRICT profile
# RULE: LOOSE profile results must be re-validated before becoming evidence

# Example: صَبْرٌ → صبر, ٱلصَّبْرِ → الصبر (STRICT)
```

---

## Graph Projection Design

### Node Types

| Node Type | Source | Count |
|-----------|--------|-------|
| BEHAVIOR | vocab/canonical_entities.json | 87 (reconciled in Phase 1) |
| VERSE | verses table | 6236 |
| TAFSIR_CHUNK | tafsir_chunks table (SSOT) | ~31,180 (5 sources × 6236) |
| AGENT | vocab/canonical_entities.json | 14 |
| ORGAN | vocab/canonical_entities.json | 40 |
| HEART_STATE | vocab/canonical_entities.json | 12 |
| CONSEQUENCE | vocab/canonical_entities.json | 16 |
| AXIS_VALUE | vocab/canonical_entities.json | 45 |

### Edge Types

| Edge Type | From → To | Description |
|-----------|-----------|-------------|
| MENTIONED_IN | BEHAVIOR → VERSE | Behavior appears in verse (with evidence_type, directness) |
| EXPLAINED_BY | VERSE → TAFSIR_CHUNK | Verse explained by tafsir (with source) |
| CAUSES | BEHAVIOR → BEHAVIOR | Causal relationship |
| LEADS_TO | BEHAVIOR → BEHAVIOR | Sequential relationship |
| PREVENTS | BEHAVIOR → BEHAVIOR | Preventive relationship |
| OPPOSITE_OF | BEHAVIOR → BEHAVIOR | Antonym relationship |
| COMPLEMENTS | BEHAVIOR → BEHAVIOR | Complementary relationship |
| STRENGTHENS | BEHAVIOR → BEHAVIOR | Reinforcement relationship |
| CONDITIONAL_ON | BEHAVIOR → BEHAVIOR | Conditional relationship |
| HAS_ROOT | BEHAVIOR → LEXEME | Behavior linked to Arabic root |
| HAS_ORGAN | BEHAVIOR → ORGAN | Behavior involves organ |
| HAS_AGENT | BEHAVIOR → AGENT | Behavior performed by agent |
| HAS_CONSEQUENCE | BEHAVIOR → CONSEQUENCE | Behavior leads to consequence |
| HAS_HEART_STATE | BEHAVIOR → HEART_STATE | Behavior associated with heart state |
| HAS_AXIS_VALUE | BEHAVIOR → AXIS_VALUE | Behavior classified on axis |

### Provenance Requirements

Every edge MUST have:
```json
{
  "provenance": {
    "verse_keys": ["2:177", "3:134"],
    "tafsir_sources": ["ibn_kathir", "qurtubi"],
    "evidence_type": "EXPLICIT",
    "confidence": 0.95,
    "extraction_method": "rule_based|ml_extracted|manual"
  }
}
```

---

## Capability Engines Mapping to 200-Question Suite

### Section A: Graph Causal / Multihop / Cycles / Spirals (25 questions)

**Engine**: `capabilities/graph_causal_engine.py`

| Capability | Questions | Implementation |
|------------|-----------|----------------|
| Path enumeration | A01, A02, A13, A18 | BFS/DFS with hop constraints |
| Cycle detection | A03, A20 | Tarjan's algorithm for A→B→C→A |
| Bottleneck/intervention | A05, A15, A22 | Betweenness centrality |
| Causal density | A06, A11 | In/out degree computation |
| Indirect causation | A07, A12 | Transitive closure analysis |
| Bidirectional detection | A08, A20 | Edge reversal check |
| Subgraph extraction | A17, A21 | Connected component analysis |
| Distance matrix | A18 | Floyd-Warshall / BFS all-pairs |
| Isolation detection | A19 | Degree-0 node detection |

### Section B: Tafsir Divergence & Consensus (25 questions)

**Engine**: `capabilities/tafsir_comparison_engine.py`

| Capability | Questions | Implementation |
|------------|-----------|----------------|
| Source fingerprinting | B01, B22 | Top-k behaviors per source |
| Verse-aligned comparison | B02, B21 | Multi-source diff on verse_key |
| Agreement metrics | B05, B11, B18 | Jaccard/overlap coefficients |
| Narrative vs legal ratio | B03 | Keyword classification |
| Linguistic depth | B04 | Morphological term density |
| Citation mapping | B06, B07 | Isnad/scholar extraction |
| Expansion factor | B09 | Word count ratios |
| Unique insights | B12 | Set difference per source |
| Temporal evolution | B16 | Classical vs modern comparison |

### Section C: 11D Axes Aggregation & Correlations (25 questions)

**Engine**: `capabilities/axes_11d_engine.py`

| Capability | Questions | Implementation |
|------------|-----------|----------------|
| Full profile generation | C01, C06 | 11-axis vector per behavior |
| Axis coverage metrics | C02 | Completeness score per axis |
| Cross-axis correlations | C03, C08 | Pearson/Spearman correlation |
| Profile similarity | C04 | Cosine similarity search |
| Outlier detection | C05 | Z-score / IQR analysis |
| Profile clustering | C07 | K-means / hierarchical |
| Axis independence | C08 | PCA / mutual information |
| Profile prediction | C09 | Regression / classification |

### Section D: Global Graph Metrics (25 questions)

**Engine**: `capabilities/graph_metrics_engine.py`

**IMPORTANT**: Use **igraph/graph-tool** for all graph metrics (NOT NetworkX).
NetworkX is too slow for 736K relations. Compute offline, store results in Postgres.

| Capability | Questions | Implementation |
|------------|-----------|----------------|
| Density/diameter | D01-D05 | **igraph** graph metrics (offline) |
| Clustering coefficient | D06-D10 | **igraph** local/global clustering (offline) |
| PageRank | D11-D15 | **igraph** PageRank (offline) |
| K-core decomposition | D16-D18 | **igraph** core number (offline) |
| Motif detection | D19-D21 | **graph-tool** subgraph isomorphism (offline) |
| Community detection | D22-D25 | **igraph** Louvain (offline) |

**Storage**: Precomputed metrics stored in `graph_metrics` table for runtime queries.

### Section E: Heart States Subgraph (20 questions)

**Engine**: `capabilities/heart_states_engine.py`

| Capability | Questions | Implementation |
|------------|-----------|----------------|
| State transition graph | E01-E10 | Markov chain from edges |
| State→behavior mapping | E11-E15 | Adjacency matrix |
| Healing pathways | E16-E20 | Shortest path to سليم |

### Section F: Agents Subgraph (20 questions)

**Engine**: `capabilities/agents_engine.py`

| Capability | Questions | Implementation |
|------------|-----------|----------------|
| Agent→behavior matrix | F01-F10 | Bipartite projection |
| Agent comparison | F11-F15 | Profile diff (مؤمن vs كافر) |
| Agent-specific patterns | F16-F20 | Subgraph extraction |

### Section G: Temporal/Spatial Mapping (15 questions)

**Engine**: `capabilities/temporal_spatial_engine.py`

| Capability | Questions | Implementation |
|------------|-----------|----------------|
| Temporal distribution | G01-G08 | دنيا/آخرة classification |
| Spatial distribution | G09-G15 | Location-based grouping |

### Section H: Consequences Subgraph (15 questions)

**Engine**: `capabilities/consequences_engine.py`

| Capability | Questions | Implementation |
|------------|-----------|----------------|
| Behavior→consequence mapping | H01-H08 | Edge traversal |
| Severity ranking | H09-H12 | Weighted scoring |
| Consequence chains | H13-H15 | Multi-hop to جنة/نار |

### Section I: Embeddings Evaluation (15 questions)

**Engine**: `capabilities/embeddings_engine.py`

| Capability | Questions | Implementation |
|------------|-----------|----------------|
| Embedding↔graph alignment | I01-I05 | Correlation analysis |
| Opposite separation | I06-I10 | Cosine distance validation |
| Category coherence | I11-I13 | Cluster purity |
| Outlier detection | I14-I15 | Isolation forest |

### Section J: Provenance & Audit (15 questions)

**Engine**: `capabilities/provenance_engine.py`

| Capability | Questions | Implementation |
|------------|-----------|----------------|
| Citation validity | J01-J05 | verse_key existence check |
| Subset contract | J06-J10 | Tafsir ⊆ verse evidence |
| Surah intro leakage | J11-J13 | entry_type filter |
| Completeness audit | J14-J15 | Coverage metrics |

---

## Evaluation Harness Specification

### Input Format

```jsonl
{"id":"A01","section":"A","title":"...","question_ar":"...","question_en":"...","expected":{...}}
```

### Expected Fields

```json
{
  "capabilities": ["GRAPH_CAUSAL", "MULTIHOP", "TAFSIR_MULTI_SOURCE", "PROVENANCE"],
  "min_sources": 2,
  "required_sources": ["ibn_kathir", "qurtubi"],
  "min_hops": 3,
  "must_include": ["edge_provenance", "verse_keys_per_link"],
  "disallow": ["generic_opening_verses_default"]
}
```

### Pass/Fail Rules

**CRITICAL**: Validation is on STRUCTURED ENGINE OUTPUT, not text length.

| Rule | Check | Fail Condition |
|------|-------|----------------|
| Engine output exists | `engine_output is not None` | No structured output |
| Required fields present | JSON schema validation | Missing required field in engine output |
| Required capabilities | All in `expected.capabilities` | Missing capability output |
| Min sources | `len(sources_used) >= min_sources` | Insufficient sources |
| Required sources | All in `expected.required_sources` | Missing required source |
| Min hops | `max_chain_length >= min_hops` | Chains too short |
| Must include | All fields present in engine output | Missing required field |
| Disallow | None present | Disallowed pattern found |
| Provenance | Every claim has verse_key in engine output | Unsupported claim |
| No fabrication | All numbers in engine output | Invented number |
| Citation validity | All verse_keys exist in SSOT | Invalid verse reference |

**Validation Flow**:
1. Engine returns structured JSON with required fields
2. Verifier validates JSON schema + citations against SSOT
3. LLM composes narrative AFTER engine output validated
4. Final verifier checks narrative doesn't add unsupported claims

### Disallow List

```python
DISALLOW_PATTERNS = [
    "generic_opening_verses_default",  # Surah 1 or 2:1-20 as fallback
    "surah_intro_as_behavior_evidence", # entry_type='surah_intro' in evidence
    "unsupported_claims",               # Claims without verse_key
    "fabricated_statistics",            # Numbers not in computed payload
    "fallback_used_for_structured_intent", # FREE_TEXT for analytics
]
```

### Output Format

```json
{
  "id": "A01",
  "status": "PASS|FAIL",
  "score": 1.0,
  "reasons": [],
  "capabilities_used": ["GRAPH_CAUSAL", "MULTIHOP"],
  "sources_used": ["ibn_kathir", "qurtubi"],
  "provenance_valid": true,
  "disallow_violations": []
}
```

---

## Audit Pack Requirements

### Input Hashes

| Asset | Path | Hash Algorithm |
|-------|------|----------------|
| Quran text | data/quran/*.jsonl | SHA256 |
| Tafsir sources | data/tafsir/*.jsonl | SHA256 |
| Canonical entities | vocab/canonical_entities.json | SHA256 |
| Postgres schema | schemas/postgres_truth_layer.sql | SHA256 |
| Embedding registry | embedding_registry table dump | SHA256 |

**NOTE**: `concept_index_v2.jsonl` and `semantic_graph_v2.json` are DEPRECATED and excluded from build inputs. They are historical artifacts only.

### Output Hashes

| Asset | Path | Hash Algorithm |
|-------|------|----------------|
| Behavior dossiers | data/kb/behavior_dossiers.jsonl | SHA256 |
| Graph edges | data/kb/graph_edges.jsonl | SHA256 |
| Embedding index | data/embeddings/*.npy | SHA256 |
| Eval report | artifacts/eval_report.json | SHA256 |

### GPU Proof Logs

```json
{
  "torch_cuda_available": true,
  "device_name": "NVIDIA A100-SXM4-40GB",
  "device_count": 1,
  "nvidia_smi_output": "...",
  "embedding_model_id": "aubmindlab/bert-base-arabertv2",
  "vector_dimensions": 768,
  "build_timestamp": "2025-12-30T19:00:00Z"
}
```

### Provenance Completeness Report

```json
{
  "total_behaviors": 87,
  "behaviors_with_verse_evidence": 87,
  "total_edges": 736,
  "edges_with_provenance": 736,
  "tafsir_coverage": {
    "ibn_kathir": 6236,
    "tabari": 6236,
    "qurtubi": 6236,
    "saadi": 6236,
    "jalalayn": 6236
  },
  "orphan_records": 0,
  "fk_violations": 0
}
```

---

## Phase-by-Phase Delivery Plan

### Phase 0: Create Implementation Plan ✅ (This Document)

**Gate**: This MD file committed  
**Commit**: `docs(plan): enterprise QBM brain implementation plan`

### Phase 1: Postgres SSOT + Arabic Normalization ✅

**Deliverables**:
- [x] Schema migration under `db/migrations/001_ssot_extensions.sql`
- [x] Arabic normalization module `src/text/ar_normalize.py` with STRICT/LOOSE profiles (v2.0.0)
- [x] Data loaders for verses, tafsir, behaviors (existing in src/data/)
- [x] **Behavior inventory reconciliation**: audit 73 vs 87, documented gaps for future reconciliation
- [x] Behavior inventory audit report: `artifacts/behavior_inventory_audit.json`

**Tests** (36 passing in `tests/test_ssot_phase1.py`):
- [x] `test_ssot_counts`: canonical entities validated, classifier has 87 behaviors
- [x] `test_normalization`: STRICT/LOOSE profiles working, صبر vs ٱلصَّبْرِ match
- [x] `test_foreign_keys`: no orphans, all IDs have correct prefixes
- [x] `test_behavior_inventory`: audit report exists with gaps documented

**Commit**: `a8d759b` - `feat(ssot): Phase 1 - Postgres schema extensions + Arabic normalization profiles`

### Phase 2: Graph Projection in Postgres ✅

**Deliverables**:
- [x] `semantic_edges` table extended with from/to entity_type + provenance JSONB (in 001_ssot_extensions.sql)
- [x] `behavior_verse_links` populated from validated sources (scripts/populate_behavior_verse_links.py)
- [x] Indexes for efficient traversal (idx_bvl_behavior, idx_bvl_verse_key)
- [x] Edge provenance populated for all edges (22,195 links, 4,460 edges updated)

**Tests** (25 passing in `tests/test_graph_phase2.py`):
- [x] `test_graph_min_edges`: behavior→verse edges exist (22,195 links created)
- [x] `test_no_surah_intro_as_behavior_evidence`: surah_intro filtering implemented
- [x] `test_subset_contract`: tafsir sources validated, verse refs valid

**Commit**: `feat(graph): Phase 2 - graph projection + behavior_verse_links + integrity tests`

### Phase 3: Precomputed Knowledge Base Build ✅

**Deliverables**:
- [x] `scripts/build_kb.py` - enhanced with SHA256 manifest generation (v2.0)
- [x] `data/kb/behavior_dossiers.jsonl` - 73 dossiers with verses, tafsir, relations
- [x] `data/kb/manifest.json` with SHA256 hashes (input + output files)

**Tests** (23 passing in `tests/test_kb_phase3.py`):
- [x] `test_kb_build_reproducible`: manifest v2.0, input/output hashes, git commit
- [x] `test_patience_anchor`: الصبر dossier exists with key verses (2:45, 2:153, 3:200, 103:3)
- [x] `test_dossier_completeness`: all 73 dossiers have required fields
- [x] `test_kb_file_integrity`: 6236 verses, 73 behaviors, valid JSONL

**Commit**: `feat(kb): Phase 3 - KB build with SHA256 manifest + reproducibility tests`

### Phase 4: Capability Engines ✅

**Deliverables**:
- [x] `src/capabilities/` module with 16 engines mapped to A-J sections
- [x] Unit tests for each engine (32 tests)
- [x] Regression tests for disallow patterns (generic opening verses filtered)

**Tests** (32 passing in `tests/test_capabilities_phase4.py`):
- [x] `test_engine_*`: deterministic tests per engine (GRAPH_CAUSAL, MULTIHOP, TAFSIR, etc.)
- [x] `test_no_generic_opening_verses`: Fatiha + early Baqarah filtered
- [x] `test_must_include_provenance`: all engines return provenance

**Commit**: `feat(engines): Phase 4 - capability engines A-J + 32 unit tests`

### Phase 5: Evaluation Harness

**Deliverables**:
- [ ] `eval/run_suite.py`
- [ ] `artifacts/eval_report.json`
- [ ] Coverage metrics by section

**Tests**:
- [ ] Full 200-question run
- [ ] Pass rate ≥180/200

**Commit**: `feat(eval): 200-question evaluation harness + reports`

### Phase 6: Azure OpenAI Integration

**Deliverables**:
- [ ] Function-calling tools (resolve_entity, get_behavior_dossier, etc.)
- [ ] Verifier gate for citation validity
- [ ] Fail-closed on violations

**Tests**:
- [ ] `test_tool_calling`: tools return expected payloads
- [ ] `test_verifier_gate`: violations caught

**Commit**: `feat(azure): tool-first orchestrator + verifier gate`

### Phase 7: Audit Pack

**Deliverables**:
- [ ] `artifacts/audit_pack/` generator
- [ ] Input/output hashes
- [ ] GPU proof logs
- [ ] Provenance completeness report

**Tests**:
- [ ] `test_audit_pack_complete`: all required files present
- [ ] `test_hashes_reproducible`: deterministic hashes

**Commit**: `feat(audit): audit pack generator + provenance checks`

---

## Acceptance Criteria Summary

| Phase | Gate | Proof Required |
|-------|------|----------------|
| 0 | Plan committed | This file exists |
| 1 | SSOT tests pass | pytest output |
| 2 | Graph tests pass | pytest output |
| 3 | KB reproducible | manifest hashes |
| 4 | Engine tests pass | pytest output |
| 5 | ≥180/200 PASS | eval_report.json |
| 6 | Tools + verifier work | pytest output |
| 7 | Audit pack complete | audit_pack/ contents |

---

## Existing Assets to Reuse (No Re-Implementation)

| Asset | Path | SSOT Status | Usage |
|-------|------|-------------|-------|
| Postgres schema | schemas/postgres_truth_layer.sql | **SSOT** | Extend, don't replace |
| Canonical entities | vocab/canonical_entities.json | **SSOT** | 87 behaviors (reconciled), 14 agents, etc. |
| Semantic graph | data/graph/semantic_graph_v2.json | **Derived** | Graph traversal (rebuilt from SSOT) |
| LegendaryPlanner | src/ml/legendary_planner.py | N/A | Question routing |
| Tafsir constants | src/ml/tafsir_constants.py | N/A | 7 canonical sources |
| Build KB script | scripts/build_kb.py | N/A | Extend for dossiers |

**DEPRECATED** (do not use as truth):
| Asset | Path | Reason |
|-------|------|--------|
| concept_index_v2.jsonl | data/evidence/concept_index_v2.jsonl | Corrupt index - replaced by `behavior_verse_links` SSOT |
| evidence_index_v2_chunked.jsonl | data/evidence/evidence_index_v2_chunked.jsonl | Derived from corrupt index - rebuild from SSOT |

---

## "Stop Wasting Time" Enforcement

From now on, "done" or "enterprise-grade" requires:

1. **Commit hash** - Proof of version control
2. **Test output** - pytest results
3. **Eval report excerpt** - Benchmark scores
4. **Manifest hashes** - Reproducibility proof

If you can't prove it, it is not done.

---

## Appendix: Canonical Tafsir Sources

```python
CANONICAL_TAFSIR_SOURCES = [
    "ibn_kathir",
    "tabari", 
    "qurtubi",
    "saadi",
    "jalalayn",
    "baghawi",
    "muyassar"
]
```

## Appendix: 11 Bouzidani Dimensions

1. **العضوي (Organic)**: القلب، اللسان، العين، الأذن، اليد، الرجل
2. **الموقفي (Situational)**: داخلي، قولي، فعلي، علائقي
3. **النظامي (Systemic)**: عبادي، أسري، مجتمعي، اقتصادي، سياسي
4. **المكاني (Spatial)**: مسجد، بيت، سوق، طريق
5. **الزماني (Temporal)**: دنيا، موت، برزخ، قيامة، آخرة
6. **الفاعل (Agent)**: مؤمن، كافر، منافق، مشرك
7. **المصدر (Source)**: وحي، فطرة، نفس، شيطان
8. **التقييم (Evaluation)**: ممدوح، مذموم، محايد
9. **حالة القلب (Heart State)**: سليم، مريض، قاسي، مختوم، ميت
10. **العاقبة (Consequence)**: دنيوية، أخروية
11. **العلاقات (Relations)**: سبب، نتيجة، نقيض، مكمل

---

## Appendix: SSOT Decision Matrix

This appendix defines the authoritative status of every artifact/table to prevent "which table is real?" confusion.

### Database Tables

| Table | SSOT Status | Built By | Validation Gate | Deprecated? |
|-------|-------------|----------|-----------------|-------------|
| `verses` | **SSOT** | Initial migration | FK constraints, count=6236 | No |
| `tafsir_chunks` | **SSOT** | `load_truth_layer_to_postgres.py` | FK to verses, entry_type check | No |
| `entities` | **SSOT** | `load_truth_layer_to_postgres.py` | Unique entity_id, type check | No |
| `mentions` | **SSOT** | `load_truth_layer_to_postgres.py` | FK to entities + tafsir_chunks | No |
| `semantic_edges` | **SSOT** | `build_semantic_graph_v2.py` | FK to entities, provenance JSONB | No |
| `edge_evidence` | **SSOT** | `build_semantic_graph_v2.py` | FK to semantic_edges + tafsir_chunks | No |
| `behavior_verse_links` | **SSOT** | Phase 1 migration | FK to entities + verses, evidence_type check | No |
| `lexemes` | **SSOT** | Phase 1 migration | Unique root+lemma+form | No |
| `verse_lexemes` | **SSOT** | Phase 1 migration | FK to verses + lexemes | No |
| `behaviors` (view) | **Derived** | Materialized view from `entities` | Refresh on entities change | No |
| `graph_metrics` | **Derived** | `build_kb.py` (igraph offline) | Rebuild on semantic_edges change | No |
| `embedding_registry` | **SSOT** | Phase 1 migration | Model provenance for embeddings | No |
| `behavior_embeddings` | **SSOT** | `build_kb.py` (GPU) | FK to entities + embedding_registry | No |
| `verse_embeddings` | **SSOT** | `build_kb.py` (GPU) | FK to verses + embedding_registry | No |
| `tafsir_embeddings` | **SSOT** | `build_kb.py` (GPU) | FK to tafsir_chunks + embedding_registry | No |

### File Artifacts

| Artifact | SSOT Status | Built By | Validation Gate | Deprecated? |
|----------|-------------|----------|-----------------|-------------|
| `vocab/canonical_entities.json` | **SSOT** | Manual curation | Schema validation, 87 behaviors | No |
| `schemas/postgres_truth_layer.sql` | **SSOT** | Manual curation | N/A | No |
| `src/ml/tafsir_constants.py` | **SSOT** | Manual curation | 7 sources list | No |
| `data/graph/semantic_graph_v2.json` | **Derived** | `build_semantic_graph_v2.py` | Rebuild from `semantic_edges` | No |
| `data/kb/behavior_dossiers.jsonl` | **Derived** | `build_kb.py` | Rebuild from SSOT tables | No |
| `artifacts/kb_manifest.json` | **Derived** | `build_kb.py` | SHA256 hashes | No |
| `data/evidence/concept_index_v2.jsonl` | **DEPRECATED** | N/A | DO NOT USE | **Yes** |
| `data/evidence/evidence_index_v2_chunked.jsonl` | **DEPRECATED** | N/A | DO NOT USE | **Yes** |

### Derivation Rules

1. **Materialized views** refresh when source SSOT changes
2. **Derived files** rebuild via `build_kb.py` with manifest hashes
3. **Deprecated artifacts** must not be read by any production code
4. **SSOT tables** are the only source for validation gates

### Migration Plan for Deprecated Assets

| Asset | Replacement | Migration Task |
|-------|-------------|----------------|
| `concept_index_v2.jsonl` | `behavior_verse_links` table | Phase 1: populate from validated sources |
| `evidence_index_v2_chunked.jsonl` | `tafsir_chunks` + `mentions` tables | Phase 1: already in SSOT |

### Validation Gate Definitions

| Gate | Check | Failure Action |
|------|-------|----------------|
| FK constraints | All foreign keys valid | Reject insert/update |
| Count checks | verses=6236, behaviors=87 | Fail build |
| Provenance check | Every edge has verse_key in provenance | Fail build |
| Evidence mode check | Every behavior has evidence_mode set | Fail build |
| Entry type check | No surah_intro in behavior evidence | Fail query |

---

**Document Status**: Ready for Phase 0 commit gate (with 7 + 6 = 13 mandatory fixes applied)

### Fixes Applied

**Round 1 (7 fixes)**:
1. Removed corrupt `concept_index_v2.jsonl` from truth assets
2. No duplicate tables - extend existing SSOT tables
3. Per-behavior evidence policy (lexical/annotation/hybrid)
4. Arabic normalization STRICT/LOOSE profiles
5. Structured engine output validation (not text length)
6. igraph/graph-tool for offline graph metrics
7. 87 behaviors reconciliation task

**Round 2 (6 fixes)**:
1. FK to `entities(entity_id)` not materialized view + trigger enforcement
2. Node types reference `tafsir_chunks` (not removed `tafsir_entries`)
3. Proper column types for roots/synonyms (TEXT[], not metadata->>'')
4. Clarified provenance (summary) vs edge_evidence (detailed rows)
5. Removed deprecated assets from audit pack input hashes
6. Added pgvector + embedding_registry + embedding tables
