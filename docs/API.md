# QBM API Reference

## Base URL

```
http://localhost:8000
```

## Authentication

Currently no authentication required. Production deployment should add API key or OAuth.

---

## Health Endpoints

### GET /api/health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-12-27T03:00:00Z"
}
```

### GET /api/ready

Readiness check with component status.

**Response:**
```json
{
  "ready": true,
  "components": {
    "quran": true,
    "tafsir": true,
    "spans": true,
    "graph": true
  }
}
```

---

## Proof Endpoints

### POST /api/proof/query

Submit a proof query and receive evidence from all layers.

**Request Body:**
```json
{
  "question": "ما هو الصبر؟",
  "include_proof": true,
  "mode": "summary",
  "per_ayah": true,
  "max_chunks_per_source": 3,
  "proof_only": false
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `question` | string | required | Query in Arabic or English |
| `include_proof` | boolean | true | Include full proof object |
| `mode` | string | "full" | "summary" or "full" |
| `per_ayah` | boolean | false | Group results by ayah |
| `max_chunks_per_source` | integer | 5 | Limit chunks per tafsir source |
| `proof_only` | boolean | false | Skip LLM answer generation for fast retrieval (v2.0.1+) |

**Query Intent Detection:**

The system automatically detects query intent:
- **SURAH_REF**: "سورة الفاتحة", "surah 1" → Full surah tafsir retrieval
- **AYAH_REF**: "2:255", "البقرة:255" → Single verse tafsir retrieval
- **CONCEPT_REF**: Behavior/concept queries → Semantic search
- **FREE_TEXT**: General queries → Hybrid retrieval

For structured intents (SURAH_REF, AYAH_REF), retrieval is deterministic from the 7-source chunked index.

**Response:**
```json
{
  "question": "ما هو الصبر؟",
  "answer": "الصبر هو...",
  "proof": {
    "quran": [
      {
        "surah": 2,
        "ayah": 155,
        "text": "وَلَنَبْلُوَنَّكُم...",
        "relevance": 0.95
      }
    ],
    "tafsir": {
      "ibn_kathir": [...],
      "tabari": [...],
      "qurtubi": [...],
      "saadi": [...],
      "jalalayn": [...],
      "baghawi": [...],
      "muyassar": [...]
    },
    "graph": {
      "nodes": [...],
      "edges": [...],
      "paths": [...]
    },
    "taxonomy": {
      "behaviors": [...],
      "dimensions": {...}
    },
    "statistics": {
      "counts": {...},
      "percentages": {...}
    }
  },
  "validation": {
    "score": 100.0,
    "passed": true,
    "missing": [],
    "checks": {...}
  },
  "debug": {
    "intent": "SURAH_REF",
    "fallback_used": false,
    "fallback_reasons": [],
    "retrieval_mode": "deterministic_chunked",
    "retrieval_distribution": {"ibn_kathir": 7, "tabari": 7, ...},
    "sources_covered": ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn", "baghawi", "muyassar"],
    "core_sources_count": 7,
    "component_fallbacks": {
      "quran": false,
      "graph": false,
      "taxonomy": false,
      "tafsir": {"ibn_kathir": false, "tabari": false, ...}
    },
    "fullpower_used": true,
    "index_source": "disk",
    "primary_path_latency_ms": 145
  },
  "processing_time_ms": 145
}
```

**proof_only Mode (v2.0.1+):**

When `proof_only=true`, the system uses a lightweight backend that:
- Does NOT initialize GPU embedding pipeline
- Does NOT call LLM for answer generation
- Uses only JSON files for deterministic retrieval
- Returns `debug.fullpower_used = false`
- Target latency: <5 seconds for structured intents

This mode is ideal for:
- Fast Tier-A tests
- Batch evidence retrieval
- Clients that generate their own answers

Response differences in `proof_only` mode:
- `answer` will be `"[proof_only mode - LLM answer skipped]"`
- `debug.fullpower_used` will be `false`
- `debug.index_source` will be `"json_chunked"`

### GET /api/proof/status

Get proof system status.

**Response:**
```json
{
  "status": "ready",
  "components": {
    "quran_loaded": true,
    "tafsir_loaded": true,
    "graph_loaded": true,
    "index_ready": true
  }
}
```

---

## Genome Endpoints

### GET /api/genome/status

Get Q25 genome status and statistics.

**Response:**
```json
{
  "status": "ready",
  "version": "1.0.0",
  "source_versions": {
    "canonical_entities": "1.0.0",
    "semantic_graph": "2.0.0"
  },
  "statistics": {
    "canonical_behaviors": 73,
    "canonical_agents": 14,
    "canonical_organs": 11,
    "canonical_heart_states": 12,
    "canonical_consequences": 16,
    "semantic_edges": 4460
  },
  "endpoints": {
    "export": "GET /api/genome/export",
    "behaviors": "GET /api/genome/behaviors",
    "agents": "GET /api/genome/agents"
  }
}
```

### GET /api/genome/export

Export the complete Q25 genome.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | string | "full" | "full" (with evidence) or "light" (metadata only) |

**Response (mode=full):**
```json
{
  "version": "1.0.0",
  "mode": "full",
  "checksum": "sha256:abc123...",
  "source_versions": {
    "canonical_entities": "1.0.0",
    "semantic_graph": "2.0.0"
  },
  "statistics": {...},
  "behaviors": [
    {
      "id": "BEH_EMO_PATIENCE",
      "name_ar": "الصبر",
      "name_en": "Patience",
      "category": "emotional",
      "evidence": [
        {
          "chunk_id": "ibn_kathir_2_155_001",
          "verse_key": "2:155",
          "char_start": 0,
          "char_end": 150,
          "quote": "..."
        }
      ]
    }
  ],
  "agents": [...],
  "organs": [...],
  "heart_states": [...],
  "consequences": [...],
  "semantic_edges": [
    {
      "source": "BEH_EMO_PATIENCE",
      "target": "CSQ_JANNAH",
      "type": "LEADS_TO",
      "confidence": 0.95,
      "evidence_count": 12
    }
  ],
  "axes": {...}
}
```

### GET /api/genome/behaviors

List all canonical behaviors.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `category` | string | null | Filter by category |
| `limit` | integer | 100 | Max results |

### GET /api/genome/agents

List all canonical agents.

### GET /api/genome/heart-states

List all heart states.

### GET /api/genome/consequences

List all consequences.

### GET /api/genome/relationships

Get semantic relationships.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | string | null | Filter by source entity |
| `target` | string | null | Filter by target entity |
| `type` | string | null | Filter by edge type |

---

## Reviews Endpoints

### GET /api/reviews/status

Get review system status.

**Response:**
```json
{
  "status": "ready",
  "backend": "sqlite",
  "statistics": {
    "total_reviews": 42,
    "by_status": {
      "pending": 15,
      "approved": 25,
      "rejected": 2
    },
    "by_type": {
      "span": 20,
      "edge": 15,
      "chunk": 7
    }
  },
  "endpoints": {...}
}
```

### GET /api/reviews

List reviews with filtering.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `status` | string | null | Filter by status (pending/approved/rejected) |
| `review_type` | string | null | Filter by type (span/edge/chunk) |
| `limit` | integer | 50 | Max results |
| `offset` | integer | 0 | Pagination offset |

**Response:**
```json
{
  "total": 42,
  "limit": 50,
  "offset": 0,
  "reviews": [
    {
      "id": 1,
      "span_id": null,
      "edge_id": "BEH_PATIENCE_CAUSES_CSQ_JANNAH",
      "chunk_id": null,
      "surah": 2,
      "ayah": 155,
      "verse_key": "2:155",
      "reviewer_id": "scholar_001",
      "reviewer_name": "Dr. Ahmad",
      "status": "approved",
      "rating": 5,
      "comment": "Evidence is strong",
      "review_type": "edge",
      "created_at": "2025-12-27T03:00:00Z",
      "updated_at": "2025-12-27T03:30:00Z"
    }
  ]
}
```

### POST /api/reviews

Create a new review.

**Request Body:**
```json
{
  "edge_id": "BEH_PATIENCE_CAUSES_CSQ_JANNAH",
  "surah": 2,
  "ayah": 155,
  "reviewer_id": "scholar_001",
  "reviewer_name": "Dr. Ahmad",
  "rating": 4,
  "comment": "Good evidence, minor clarification needed"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `span_id` | string | no | Span ID to review |
| `edge_id` | string | no | Edge ID to review |
| `chunk_id` | string | no | Chunk ID to review |
| `surah` | integer | no | Surah number |
| `ayah` | integer | no | Ayah number |
| `reviewer_id` | string | yes | Reviewer identifier |
| `reviewer_name` | string | no | Reviewer display name |
| `rating` | integer | no | 1-5 rating |
| `comment` | string | no | Review comment |

**Note:** At least one of `span_id`, `edge_id`, or `chunk_id` should be provided.

### GET /api/reviews/{id}

Get a specific review with history.

**Response:**
```json
{
  "id": 1,
  "edge_id": "BEH_PATIENCE_CAUSES_CSQ_JANNAH",
  "status": "approved",
  "history": [
    {
      "action": "status_change",
      "old_status": "pending",
      "new_status": "approved",
      "actor_id": "admin_001",
      "timestamp": "2025-12-27T03:30:00Z"
    }
  ]
}
```

### PUT /api/reviews/{id}

Update a review.

**Query Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `actor_id` | string | yes | ID of actor making the change |

**Request Body:**
```json
{
  "status": "approved",
  "comment": "Updated comment"
}
```

### DELETE /api/reviews/{id}

Delete a review.

**Query Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `actor_id` | string | yes | ID of actor making the deletion |

### GET /api/reviews/edge/{edge_id}

Get all reviews for a specific edge.

### GET /api/reviews/chunk/{chunk_id}

Get all reviews for a specific chunk.

### GET /api/reviews/export

Export reviews as JSON.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `status` | string | null | Filter by status |
| `review_type` | string | null | Filter by type |

---

## Graph Endpoints

### GET /api/graph/status

Get graph system status.

### GET /api/graph/verse/{surah}/{ayah}

Get graph data for a specific verse.

### GET /api/graph/behavior/{behavior}

Get graph data for a behavior.

### GET /api/graph/traverse/{node_id}

Traverse the graph from a node.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `depth` | integer | 2 | Traversal depth |
| `direction` | string | "both" | "outgoing", "incoming", or "both" |

---

## Quran Endpoints

### GET /api/datasets/{tier}

Get dataset by tier (silver, gold, platinum).

### GET /api/spans

Search behavioral spans.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `q` | string | null | Search query |
| `surah` | integer | null | Filter by surah |
| `agent` | string | null | Filter by agent type |
| `limit` | integer | 50 | Max results |
| `offset` | integer | 0 | Pagination offset |

### GET /api/surahs

List all surahs with statistics.

### GET /api/ayah/{surah}/{ayah}

Get a specific ayah with annotations.

---

## Tafsir Endpoints

### GET /tafsir/{surah}/{ayah}

Get tafsir for a verse.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | string | null | Specific tafsir source (ibn_kathir, tabari, qurtubi, saadi, jalalayn, baghawi, muyassar) |

### GET /tafsir/compare/{surah}/{ayah}

Compare tafsir from all 7 sources.

---

## Pagination

All list endpoints support pagination:

```json
{
  "total": 1000,
  "limit": 50,
  "offset": 0,
  "items": [...]
}
```

**Deterministic Ordering:**
- Results are sorted with tie-breakers to prevent duplicates across pages
- Tie-breakers: score → surah → ayah → source → chunk_id

---

## Error Responses

All errors follow this format:

```json
{
  "detail": "Error message here",
  "status_code": 400
}
```

| Status Code | Meaning |
|-------------|---------|
| 400 | Bad request (invalid parameters) |
| 404 | Resource not found |
| 422 | Validation error |
| 500 | Internal server error |

---

## Rate Limiting

No rate limiting in development. Production should implement:
- 100 requests/minute for proof queries
- 1000 requests/minute for read-only endpoints
