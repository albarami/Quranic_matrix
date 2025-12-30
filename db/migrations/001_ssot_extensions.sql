-- ============================================================================
-- Migration 001: SSOT Extensions for Enterprise QBM Brain
-- Phase 1: Postgres SSOT schema extensions
-- 
-- This migration extends the existing postgres_truth_layer.sql schema with:
-- 1. Evidence policy columns on entities table
-- 2. behavior_verse_links table with FK to entities (not materialized view)
-- 3. Extended tafsir_chunks with entry_type
-- 4. Lexeme tables for root/lemma tracking
-- 5. Extended semantic_edges with provenance
-- 6. pgvector extension and embedding tables
-- 
-- Run after: schemas/postgres_truth_layer.sql
-- ============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================================
-- 1. Add evidence policy columns to entities table (the SSOT)
-- Store as proper columns, not in metadata JSONB (faster, validated)
-- ============================================================================

ALTER TABLE entities ADD COLUMN IF NOT EXISTS evidence_mode VARCHAR(20) 
    CHECK (evidence_mode IN ('lexical', 'annotation', 'hybrid')) DEFAULT 'hybrid';

ALTER TABLE entities ADD COLUMN IF NOT EXISTS lexical_required BOOLEAN DEFAULT false;

ALTER TABLE entities ADD COLUMN IF NOT EXISTS allowed_evidence_types TEXT[] 
    DEFAULT ARRAY['EXPLICIT', 'IMPLICIT', 'CONTEXTUAL', 'TAFSIR_DERIVED'];

ALTER TABLE entities ADD COLUMN IF NOT EXISTS roots TEXT[];

ALTER TABLE entities ADD COLUMN IF NOT EXISTS synonyms TEXT[];

-- ============================================================================
-- 2. Materialized view for convenience queries (reads from columns)
-- ============================================================================

DROP MATERIALIZED VIEW IF EXISTS behaviors;

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

CREATE UNIQUE INDEX IF NOT EXISTS idx_behaviors_id ON behaviors(behavior_id);

-- ============================================================================
-- 3. behavior_verse_links (explicit behavior→verse evidence)
-- NOTE: FK references entities(entity_id), NOT the materialized view
-- Postgres cannot enforce FK against materialized views
-- ============================================================================

CREATE TABLE IF NOT EXISTS behavior_verse_links (
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
    
    FOREIGN KEY (behavior_id) REFERENCES entities(entity_id) ON DELETE CASCADE,
    FOREIGN KEY (surah, ayah) REFERENCES verses(surah, ayah) ON DELETE CASCADE,
    UNIQUE(behavior_id, surah, ayah, evidence_type)
);

CREATE INDEX IF NOT EXISTS idx_bvl_behavior ON behavior_verse_links(behavior_id);
CREATE INDEX IF NOT EXISTS idx_bvl_verse_key ON behavior_verse_links(verse_key);

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

DROP TRIGGER IF EXISTS enforce_behavior_type ON behavior_verse_links;
CREATE TRIGGER enforce_behavior_type
    BEFORE INSERT OR UPDATE ON behavior_verse_links
    FOR EACH ROW EXECUTE FUNCTION check_behavior_entity_type();

-- ============================================================================
-- 4. Extend tafsir_chunks (SSOT) with entry_type
-- DO NOT create tafsir_entries duplicate
-- ============================================================================

ALTER TABLE tafsir_chunks ADD COLUMN IF NOT EXISTS entry_type VARCHAR(20) 
    CHECK (entry_type IN ('verse', 'surah_intro', 'appendix')) DEFAULT 'verse';

ALTER TABLE tafsir_chunks ADD COLUMN IF NOT EXISTS text_normalized TEXT;

CREATE INDEX IF NOT EXISTS idx_chunks_entry_type ON tafsir_chunks(entry_type);

-- ============================================================================
-- 5. Lexemes table (roots/lemmas/forms)
-- ============================================================================

CREATE TABLE IF NOT EXISTS lexemes (
    id SERIAL PRIMARY KEY,
    root VARCHAR(20) NOT NULL,
    lemma VARCHAR(50),
    form VARCHAR(50),
    pos VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(root, lemma, form)
);

CREATE INDEX IF NOT EXISTS idx_lexemes_root ON lexemes(root);

-- ============================================================================
-- 6. Verse lexemes (verse→lexeme mapping)
-- ============================================================================

CREATE TABLE IF NOT EXISTS verse_lexemes (
    id SERIAL PRIMARY KEY,
    surah INTEGER NOT NULL,
    ayah INTEGER NOT NULL,
    verse_key VARCHAR(10) GENERATED ALWAYS AS (surah || ':' || ayah) STORED,
    lexeme_id INTEGER NOT NULL,
    word_position INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (surah, ayah) REFERENCES verses(surah, ayah) ON DELETE CASCADE,
    FOREIGN KEY (lexeme_id) REFERENCES lexemes(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_verse_lexemes_verse ON verse_lexemes(verse_key);
CREATE INDEX IF NOT EXISTS idx_verse_lexemes_lexeme ON verse_lexemes(lexeme_id);

-- ============================================================================
-- 7. Extend semantic_edges (SSOT) with entity types and provenance
-- DO NOT create graph_edges duplicate
-- ============================================================================

ALTER TABLE semantic_edges ADD COLUMN IF NOT EXISTS from_entity_type VARCHAR(30);
ALTER TABLE semantic_edges ADD COLUMN IF NOT EXISTS to_entity_type VARCHAR(30);

-- provenance column = HIGH-LEVEL SUMMARY (counts, sources list)
-- edge_evidence table = DETAILED EVIDENCE ROWS (verse_keys, tafsir refs, spans)
ALTER TABLE semantic_edges ADD COLUMN IF NOT EXISTS provenance JSONB;

COMMENT ON COLUMN semantic_edges.provenance IS 
    'High-level provenance summary: {sources_list, evidence_count, confidence}. Detailed evidence in edge_evidence table.';

-- Update entity types from entities table
UPDATE semantic_edges se SET 
    from_entity_type = (SELECT entity_type FROM entities WHERE entity_id = se.from_entity_id),
    to_entity_type = (SELECT entity_type FROM entities WHERE entity_id = se.to_entity_id)
WHERE from_entity_type IS NULL OR to_entity_type IS NULL;

-- ============================================================================
-- 8. Embedding registry (model provenance for zero-hallucination)
-- ============================================================================

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

-- ============================================================================
-- 9. Behavior embeddings table
-- ============================================================================

CREATE TABLE IF NOT EXISTS behavior_embeddings (
    id SERIAL PRIMARY KEY,
    behavior_id VARCHAR(50) NOT NULL,
    model_id VARCHAR(100) NOT NULL,
    embedding vector(768),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (behavior_id) REFERENCES entities(entity_id) ON DELETE CASCADE,
    FOREIGN KEY (model_id) REFERENCES embedding_registry(model_id) ON DELETE CASCADE,
    UNIQUE(behavior_id, model_id)
);

-- ============================================================================
-- 10. Verse embeddings table
-- ============================================================================

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

-- ============================================================================
-- 11. Tafsir chunk embeddings table
-- ============================================================================

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

-- ============================================================================
-- 12. Graph metrics table (precomputed from igraph offline)
-- ============================================================================

CREATE TABLE IF NOT EXISTS graph_metrics (
    id SERIAL PRIMARY KEY,
    entity_id VARCHAR(50) NOT NULL,
    metric_type VARCHAR(50) NOT NULL,
    metric_value REAL NOT NULL,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (entity_id) REFERENCES entities(entity_id) ON DELETE CASCADE,
    UNIQUE(entity_id, metric_type)
);

CREATE INDEX IF NOT EXISTS idx_graph_metrics_entity ON graph_metrics(entity_id);
CREATE INDEX IF NOT EXISTS idx_graph_metrics_type ON graph_metrics(metric_type);

-- ============================================================================
-- 13. Update metadata
-- ============================================================================

INSERT INTO truth_layer_metadata (key, value) VALUES
    ('migration_001_applied', CURRENT_TIMESTAMP::TEXT),
    ('schema_version', '2.0'),
    ('behaviors_count_target', '87'),
    ('normalization_profiles', 'STRICT,LOOSE')
ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = CURRENT_TIMESTAMP;

-- ============================================================================
-- Verification queries (run after migration)
-- ============================================================================

-- SELECT COUNT(*) as entity_count FROM entities WHERE entity_type = 'BEHAVIOR';
-- SELECT COUNT(*) as behavior_view_count FROM behaviors;
-- SELECT * FROM truth_layer_metadata ORDER BY key;
