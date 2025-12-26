-- QBM Truth Layer PostgreSQL Schema (Phase 8.4)
-- Single source of truth for enterprise-grade traceability
-- 
-- Tables:
--   verses         - Quran verses with metadata
--   tafsir_chunks  - Chunked tafsir evidence with offsets
--   entities       - Canonical entities (behaviors, agents, organs, etc.)
--   mentions       - Entity mentions in tafsir with provenance
--   semantic_edges - Typed relationships between entities
--   edge_evidence  - Evidence supporting each edge

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================
-- VERSES TABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS verses (
    id SERIAL PRIMARY KEY,
    surah INTEGER NOT NULL CHECK (surah >= 1 AND surah <= 114),
    ayah INTEGER NOT NULL CHECK (ayah >= 1),
    verse_key VARCHAR(10) GENERATED ALWAYS AS (surah || ':' || ayah) STORED,
    text_uthmani TEXT NOT NULL,
    text_simple TEXT,
    makki_madani VARCHAR(10) CHECK (makki_madani IN ('makki', 'madani', 'disputed')),
    juz INTEGER CHECK (juz >= 1 AND juz <= 30),
    hizb INTEGER CHECK (hizb >= 1 AND hizb <= 60),
    word_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(surah, ayah)
);

CREATE INDEX idx_verses_surah ON verses(surah);
CREATE INDEX idx_verses_verse_key ON verses(verse_key);
CREATE INDEX idx_verses_makki_madani ON verses(makki_madani);

-- ============================================================
-- TAFSIR_CHUNKS TABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS tafsir_chunks (
    id SERIAL PRIMARY KEY,
    chunk_id VARCHAR(100) UNIQUE NOT NULL,
    source VARCHAR(50) NOT NULL,  -- e.g., 'ibn_kathir', 'tabari'
    surah INTEGER NOT NULL,
    ayah INTEGER NOT NULL,
    verse_key VARCHAR(10) GENERATED ALWAYS AS (surah || ':' || ayah) STORED,
    chunk_index INTEGER DEFAULT 0,  -- For multi-chunk verses
    text_clean TEXT NOT NULL,
    text_original TEXT,
    char_start INTEGER DEFAULT 0,
    char_end INTEGER,
    word_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (surah, ayah) REFERENCES verses(surah, ayah) ON DELETE CASCADE
);

CREATE INDEX idx_chunks_source ON tafsir_chunks(source);
CREATE INDEX idx_chunks_verse_key ON tafsir_chunks(verse_key);
CREATE INDEX idx_chunks_surah ON tafsir_chunks(surah);
CREATE INDEX idx_chunks_chunk_id ON tafsir_chunks(chunk_id);

-- ============================================================
-- ENTITIES TABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS entities (
    id SERIAL PRIMARY KEY,
    entity_id VARCHAR(50) UNIQUE NOT NULL,  -- e.g., 'BEH_EMO_ENVY'
    entity_type VARCHAR(20) NOT NULL CHECK (entity_type IN (
        'BEHAVIOR', 'AGENT', 'ORGAN', 'HEART_STATE', 'CONSEQUENCE', 'AXIS_VALUE'
    )),
    label_ar VARCHAR(100) NOT NULL,
    label_en VARCHAR(100),
    category VARCHAR(50),  -- e.g., 'emotional', 'spiritual'
    polarity VARCHAR(20),  -- 'positive', 'negative', 'neutral'
    temporal VARCHAR(20),  -- 'dunya', 'akhira', 'both'
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_entities_type ON entities(entity_type);
CREATE INDEX idx_entities_entity_id ON entities(entity_id);
CREATE INDEX idx_entities_label_ar ON entities(label_ar);

-- ============================================================
-- MENTIONS TABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS mentions (
    id SERIAL PRIMARY KEY,
    entity_id VARCHAR(50) NOT NULL,
    source VARCHAR(50) NOT NULL,
    surah INTEGER NOT NULL,
    ayah INTEGER NOT NULL,
    verse_key VARCHAR(10) GENERATED ALWAYS AS (surah || ':' || ayah) STORED,
    chunk_id VARCHAR(100) NOT NULL,
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL,
    quote TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (entity_id) REFERENCES entities(entity_id) ON DELETE CASCADE,
    FOREIGN KEY (chunk_id) REFERENCES tafsir_chunks(chunk_id) ON DELETE CASCADE
);

CREATE INDEX idx_mentions_entity ON mentions(entity_id);
CREATE INDEX idx_mentions_source ON mentions(source);
CREATE INDEX idx_mentions_verse_key ON mentions(verse_key);
CREATE INDEX idx_mentions_chunk ON mentions(chunk_id);

-- ============================================================
-- SEMANTIC_EDGES TABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS semantic_edges (
    id SERIAL PRIMARY KEY,
    edge_id UUID DEFAULT uuid_generate_v4() UNIQUE,
    from_entity_id VARCHAR(50) NOT NULL,
    to_entity_id VARCHAR(50) NOT NULL,
    edge_type VARCHAR(30) NOT NULL CHECK (edge_type IN (
        'CAUSES', 'LEADS_TO', 'PREVENTS', 'OPPOSITE_OF', 
        'COMPLEMENTS', 'CONDITIONAL_ON', 'STRENGTHENS'
    )),
    confidence REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    evidence_count INTEGER DEFAULT 0,
    sources_count INTEGER DEFAULT 0,
    cue_strength VARCHAR(10) CHECK (cue_strength IN ('strong', 'medium', 'weak')),
    cue_phrases TEXT[],  -- Array of cue phrases used
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (from_entity_id) REFERENCES entities(entity_id) ON DELETE CASCADE,
    FOREIGN KEY (to_entity_id) REFERENCES entities(entity_id) ON DELETE CASCADE,
    
    UNIQUE(from_entity_id, to_entity_id, edge_type)
);

CREATE INDEX idx_edges_from ON semantic_edges(from_entity_id);
CREATE INDEX idx_edges_to ON semantic_edges(to_entity_id);
CREATE INDEX idx_edges_type ON semantic_edges(edge_type);
CREATE INDEX idx_edges_confidence ON semantic_edges(confidence DESC);

-- ============================================================
-- EDGE_EVIDENCE TABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS edge_evidence (
    id SERIAL PRIMARY KEY,
    edge_id UUID NOT NULL,
    source VARCHAR(50) NOT NULL,
    surah INTEGER NOT NULL,
    ayah INTEGER NOT NULL,
    verse_key VARCHAR(10) GENERATED ALWAYS AS (surah || ':' || ayah) STORED,
    chunk_id VARCHAR(100) NOT NULL,
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL,
    quote TEXT NOT NULL,
    cue_phrase VARCHAR(50),
    endpoints_validated BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (edge_id) REFERENCES semantic_edges(edge_id) ON DELETE CASCADE,
    FOREIGN KEY (chunk_id) REFERENCES tafsir_chunks(chunk_id) ON DELETE CASCADE
);

CREATE INDEX idx_edge_evidence_edge ON edge_evidence(edge_id);
CREATE INDEX idx_edge_evidence_source ON edge_evidence(source);
CREATE INDEX idx_edge_evidence_verse ON edge_evidence(verse_key);

-- ============================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================

-- Entity mention counts by source
CREATE OR REPLACE VIEW entity_source_coverage AS
SELECT 
    e.entity_id,
    e.entity_type,
    e.label_ar,
    m.source,
    COUNT(*) as mention_count
FROM entities e
LEFT JOIN mentions m ON e.entity_id = m.entity_id
GROUP BY e.entity_id, e.entity_type, e.label_ar, m.source
ORDER BY e.entity_id, m.source;

-- Edge summary with evidence counts
CREATE OR REPLACE VIEW edge_summary AS
SELECT 
    se.edge_id,
    se.from_entity_id,
    e1.label_ar as from_label,
    se.to_entity_id,
    e2.label_ar as to_label,
    se.edge_type,
    se.confidence,
    se.evidence_count,
    se.sources_count,
    se.cue_strength
FROM semantic_edges se
JOIN entities e1 ON se.from_entity_id = e1.entity_id
JOIN entities e2 ON se.to_entity_id = e2.entity_id
ORDER BY se.confidence DESC;

-- ============================================================
-- INTEGRITY CONSTRAINTS
-- ============================================================

-- Ensure edge evidence references valid edges
ALTER TABLE edge_evidence 
ADD CONSTRAINT fk_edge_evidence_edge 
FOREIGN KEY (edge_id) REFERENCES semantic_edges(edge_id) ON DELETE CASCADE;

-- ============================================================
-- METADATA TABLE FOR VERSIONING
-- ============================================================
CREATE TABLE IF NOT EXISTS truth_layer_metadata (
    id SERIAL PRIMARY KEY,
    key VARCHAR(100) UNIQUE NOT NULL,
    value TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO truth_layer_metadata (key, value) VALUES
    ('schema_version', '1.0'),
    ('created_at', CURRENT_TIMESTAMP::TEXT),
    ('canonical_entities_version', '1.0'),
    ('concept_index_version', '2.0'),
    ('semantic_graph_version', '2.0')
ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = CURRENT_TIMESTAMP;
