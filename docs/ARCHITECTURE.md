# QBM Architecture

## Overview

The Quranic Behavioral Matrix (QBM) is a multi-layer evidence retrieval and proof system for Quranic behavioral research. It combines deterministic truth layers with semantic search to provide auditable, evidence-backed answers.

## System Layers

```
┌─────────────────────────────────────────────────────────────┐
│                      Frontend (Next.js)                      │
│  Proof Page │ Explorer │ Genome │ Reviews │ Discovery        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      API Layer (FastAPI)                     │
│  /api/proof │ /api/genome │ /api/reviews │ /api/graph        │
└─────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   Truth Layer   │ │  Semantic Layer │ │   Graph Layer   │
│  (Deterministic)│ │   (Embeddings)  │ │  (Relationships)│
└─────────────────┘ └─────────────────┘ └─────────────────┘
          │                   │                   │
          ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────────────┐
│                      Data Layer                              │
│  Quran │ Tafsir (5 sources) │ Spans │ Canonical Entities     │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Truth Layer (Deterministic)

The foundation of QBM. All evidence must trace back to canonical sources.

**Components:**
- `vocab/canonical_entities.json` - 73 behaviors, 14 agents, 11 organs, 12 heart states, 16 consequences
- `data/graph/semantic_graph_v2.json` - 4460 evidence-backed semantic edges
- `data/quran/quran_uthmani.json` - Complete Quran text
- `data/tafsir/*.jsonl` - 7 classical tafsir sources

**Key Principle:** No fabrication. Every claim must have provenance (chunk_id, verse_key, char_start/end).

### 2. Retrieval Layer

Hybrid retrieval combining deterministic and semantic approaches.

**Query Types:**
- `SURAH_REF` - Direct surah/ayah lookup
- `AYAH_REF` - Specific verse reference
- `CONCEPT_REF` - Behavior/concept search
- `FREE_TEXT` - Semantic search fallback

**Retrieval Pipeline:**
1. Query classification (deterministic routing)
2. BM25 keyword search
3. FAISS vector search (when available)
4. Result fusion and deduplication
5. Deterministic ordering with tie-breakers

### 3. Graph Layer

Semantic relationships between entities.

**Edge Types:**
- `CAUSES` - Causal relationship
- `LEADS_TO` - Sequential relationship
- `PREVENTS` - Preventive relationship
- `ASSOCIATED_WITH` - Co-occurrence
- `OPPOSITE_OF` - Antonym relationship

**Graph Sources:**
- `semantic_graph_v2.json` - 4460 edges with confidence scores
- Evidence provenance for each edge

### 4. Proof System

Assembles evidence from all layers into a validated proof response.

**Proof Components:**
- Quran verses (with relevance scores)
- Tafsir chunks (7 sources)
- Graph paths (nodes, edges, traversals)
- Taxonomy classification (11 axes)
- Statistics and validation

**Validation:**
- All 13 components must be present
- No fabricated evidence
- Checksum for reproducibility

## API Routers

| Router | Prefix | Purpose |
|--------|--------|---------|
| `health.py` | `/api` | Health checks |
| `proof.py` | `/api/proof` | Query and proof generation |
| `quran.py` | `/api` | Quran data access |
| `tafsir.py` | `/tafsir` | Tafsir comparison |
| `genome.py` | `/api/genome` | Q25 genome export |
| `reviews.py` | `/api/reviews` | Scholar review workflow |
| `graph.py` | `/api/graph` | Graph traversal |
| `concepts.py` | `/api` | Behavior analysis |

## Data Flow

### Proof Query Flow

```
User Query
    │
    ▼
┌─────────────────┐
│ Query Classifier│ → Determines query type (SURAH_REF, CONCEPT_REF, etc.)
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Retrieval Layer │ → BM25 + FAISS hybrid search
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Evidence Filter │ → Removes placeholders, validates provenance
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Graph Enrichment│ → Adds semantic edges and paths
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Proof Assembly  │ → Combines all evidence layers
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Validation      │ → Checks completeness, no fabrication
└─────────────────┘
    │
    ▼
Proof Response (JSON)
```

### Genome Export Flow

```
GET /api/genome/export?mode=full
    │
    ▼
┌─────────────────┐
│ Load Canonical  │ → canonical_entities.json
│ Entities        │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Load Semantic   │ → semantic_graph_v2.json
│ Graph           │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Build Artifact  │ → Assemble genome with provenance
└─────────────────┘
    │
    ▼
┌─────────────────┐
│ Compute Checksum│ → SHA256 for reproducibility
└─────────────────┘
    │
    ▼
Genome JSON (with checksum)
```

## Frontend Architecture

### Pages

| Page | Route | Purpose |
|------|-------|---------|
| Home | `/` | Landing page with stats |
| Proof | `/proof` | Full proof query interface |
| Explorer | `/explorer` | Surah browser with graph |
| Genome | `/genome` | Q25 genome export |
| Reviews | `/reviews` | Scholar review workflow |
| Discovery | `/discovery` | Semantic exploration |
| Dashboard | `/dashboard` | Analytics |

### API Client

`src/lib/api.ts` provides:
- Zod schemas for response validation
- Type-safe API functions
- Error handling

### State Management

- React hooks for local state
- Zustand for global state (if needed)
- No Redux (keeping it simple)

## Security

- No hardcoded API keys
- Parameterized SQL queries (SQLite with ? placeholders)
- Input validation with Pydantic
- CORS configured for frontend origin

## Performance

### Targets

| Stage | Target |
|-------|--------|
| Stage A (Retrieval) | <50ms |
| Stage B (Graph) | <60ms |
| Stage C (Assembly) | <40ms |
| Total | <150ms |

### Optimizations

- Deterministic ordering with tie-breakers (no pagination duplicates)
- Summary mode for large result sets
- Lazy loading of full evidence
- FAISS index for vector search

## Testing

### Backend Tests

```bash
pytest tests/ -v
```

- `test_api.py` - 21 API endpoint tests
- `test_genome_q25.py` - 23 genome export tests
- `test_pagination_deterministic.py` - 8 pagination tests

### Frontend Tests

```bash
cd qbm-frontendv3
npm run test:e2e
```

- Playwright E2E tests for all pages
- Navigation and language toggle tests

## Deployment

See `docs/DEPLOYMENT.md` for:
- Environment variables
- Docker configuration
- Production checklist
