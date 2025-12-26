# QBM Legendary 25 Acceptance Specification

**Purpose**: Validate that QBM can answer deep queries with accuracy, explainability, provenance, and cross-tafsir grounding—not just "a nice answer."

---

## 0) Global Invariants (must hold for every question)

### I0. No Fabrication
- If evidence is missing: return `status="no_evidence"` for that component.
- Never synthesize verses, tafsir quotes, graph edges, or behaviors.

### I1. Evidence Provenance
Every cited tafsir quote must include:
- `source` (mufassir ID)
- `verse_key` (surah:ayah)
- `chunk_id`
- `char_start`, `char_end`
- `quote` that actually contains the referenced endpoint terms (post-normalization).

### I2. Graph Correctness
- **Co-occurrence graph** = discovery only. Never used for causal chains.
- **Semantic/claim graph** edges must include evidence offsets + validated endpoints + (for causal types) cue phrase present in the same quote span.

### I3. Deterministic Routing for Structured Queries
| Intent | Behavior |
|--------|----------|
| `AYAH_REF` | Deterministic lookup (all 5 core sources by design) |
| `SURAH_REF` | Deterministic enumeration of all ayat + core sources (no silent truncation) |
| `CONCEPT_REF` | Deterministic concept index first, hybrid only as recall booster |
| `FREE_TEXT` | Hybrid retrieval, with explicit "best effort" label |

### I4. Stable Response Contract
Response must always contain:
- `answer` (narrative)
- `proof` (structured evidence objects)
- `debug` (plan + routing + coverage + fallbacks)
- `validation` (optional but recommended)

### I5. Coverage Rules
- "≥3 mufassirin" means: at least 3 distinct sources supporting that link/claim.
- "5 sources" means the 5 core sources: Ibn Kathir, Tabari, Qurtubi, Saadi, Jalalayn.

---

## 1) Standard Output Requirements

### 1.1 Proof Bundle Structure (minimum)

```json
{
  "proof": {
    "quran": {
      "verses": [{"surah": 2, "ayah": 102, "text": "...", "relevance": "primary"}]
    },
    "tafsir": {
      "ibn_kathir": {"quotes": [{"chunk_id": "...", "char_start": 0, "char_end": 100, "quote": "..."}]},
      "tabari": {"quotes": [...]},
      "qurtubi": {"quotes": [...]},
      "saadi": {"quotes": [...]},
      "jalalayn": {"quotes": [...]}
    },
    "graph": {
      "nodes": [],
      "edges": [],
      "paths": []
    },
    "taxonomy": {
      "behaviors": [],
      "status": "found|no_evidence"
    }
  },
  "debug": {
    "query_intent": "CONCEPT_REF",
    "question_class": "causal_chain",
    "plan_steps": [],
    "core_sources_count": 5,
    "sources_covered": ["ibn_kathir", "tabari", "qurtubi", "saadi", "jalalayn"]
  }
}
```

### 1.2 Determinism Assertions (for enterprise)
Running the same query twice must return:
- Identical verse keys and chunk IDs in primary evidence sections
- Ordering may differ only if explicitly documented (e.g., equal BM25 scores)

---

## 2) Acceptance Checklist per Question (Q1–Q25)

Each question has:
- **Required components**
- **Required outputs**
- **Pass/Fail assertions** (automatable)

---

### Category 1: Causal Chain Analysis

#### Q1 — Path to Destruction (غفلة → كفر)

**Must do:**
- Use semantic claim graph only.
- Produce ≥1 causal chain from غفلة to كفر.

**Evidence requirements:**
- For each edge in the chain:
  - `edge_type` ∈ {CAUSES, LEADS_TO, STRENGTHENS, CONDITIONAL_ON}
  - ≥3 distinct tafsir sources support the edge OR mark edge `confidence < threshold` + `insufficient_support`.

**Pass:**
- `debug.question_class == "causal_chain"`
- Every chain edge has evidence offsets and validated endpoints (I1/I2).
- No co-occurrence edges in chain.

**Fail:**
- Any edge missing evidence offsets
- Any chain uses `CO_OCCURS_WITH`
- Any "causal" claim without ≥3 sources and without being flagged as insufficient

---

#### Q2 — Shortest transformation path (كبر → تواضع)

**Must do:**
- Use semantic graph shortest path algorithm.
- Return intermediate "bridge behaviors."

**Pass:**
- Returns a path with:
  - `start=كبر` concept_id, `end=تواضع` concept_id
  - list of intermediates (≥1 unless direct opposite edge exists)
  - Each hop has evidence (I1/I2) OR explicitly flagged as low-confidence.

---

#### Q3 — Reinforcement loops (A strengthens B strengthens C strengthens A)

**Must do:**
- Detect cycles using semantic STRENGTHENS edges only.

**Pass:**
- Each loop shows:
  - nodes A,B,C
  - edges with evidence and verse/tafsir citations

**Fail:**
- Loop is built from co-occurrence edges
- Edge lacks evidence offsets

---

### Category 2: Cross-Tafsir Comparative Analysis

#### Q4 — Methodological divergence (ربا across occurrences)

**Must do:**
- Enumerate all verse occurrences of ربا (deterministic verse set).
- Retrieve tafsir for each occurrence from Ibn Kathir, Qurtubi, Jalalayn at minimum.

**Pass:**
- Output includes:
  - occurrence list (verse_keys)
  - per-source comparison with methodology tag (from tafsir metadata)
  - explicit agreement vs emphasis differences
- No truncation without pagination or "summary mode."

---

#### Q5 — Makki vs Madani patience + classical vs modern

**Must do:**
- Split patience occurrences by Makki/Madani (must have verse classification data).
- Compare Tabari vs Saadi contextual framing.

**Pass:**
- Two distributions: Makki and Madani.
- Evidence samples from each group.
- Notes differences labeled "methodology/era effect" where appropriate.

---

#### Q6 — Consensus vs dispute across 5 tafsir sources

**Must do:**
- Define a measurable notion of "agreement" (annotation agreement, mention alignment, or classification agreement).
- Quantify agreement rates.

**Pass:**
- Output includes:
  - list of concepts with high agreement
  - list with disagreements
  - metrics + how computed in debug trace

---

### Category 3: 11-Axis + Taxonomy

#### Q7 — Full 11D profile for الحسد

**Must do:**
- Resolve حسد to canonical behavior concept_id.
- Produce 11-axis profile with evidence.

**Pass:**
- Axis fields present (even if "unknown" with explanation).
- 5 core sources coverage in tafsir evidence.
- Organ + agent + consequences references are typed correctly (no conflation).

---

#### Q8 — Organ-behavior mapping

**Must do:**
- For each organ entity (ORG_*), list behaviors associated via evidence.
- Provide counts and clusters.

**Pass:**
- No organ treated as behavior.
- Cluster output must be reproducible and based on defined algorithm (community detection on a defined graph).

---

#### Q9 — Heart journey (قلب سليم → قلب قاس)

**Must do:**
- Model as state transitions using semantic edges with evidence.

**Pass:**
- Returns a chain of heart states + causing behaviors.
- Each transition has evidence offsets and verse citations.

---

### Category 4: Agent-Based Analysis

#### Q10 — Divine vs human attribution

**Must do:**
- Distinguish divine فعل/صفة فعل vs human مكلف actions.
- Use agent taxonomy + theological framing note.

**Pass:**
- Output explicitly separates "attribution types."
- No mixing Allah as "behavior performer" without theological framing field.

---

#### Q11 — Believer vs disbeliever contrast matrix

**Must do:**
- Construct a matrix: agent class × behaviors × consequences.

**Pass:**
- Matrix shows exclusive/shared behaviors + evidence.
- Consequences are evidenced.

---

#### Q12 — Prophetic behavioral archetype

**Must do:**
- Extract prophet-attributed behaviors with verse evidence.
- Show frequency + prophet-specific uniqueness.

**Pass:**
- Prophet entity types are agents/roles, not behaviors.
- Includes prophet name attribution evidence.

---

### Category 5: Network + Graph Analytics

#### Q13 — Behavioral centrality top 5

**Must do:**
- Compute centrality on a defined graph (semantic graph recommended).
- Specify metric (degree/betweenness/pagerank).

**Pass:**
- Returns top 5 with metrics + explanation.
- Graph version + node counts included in debug.

---

#### Q14 — Community detection clusters

**Must do:**
- Run a deterministic community algorithm with fixed seed.
- Provide cluster names derived from cluster characteristics.

**Pass:**
- Cluster membership reproducible.
- Evidence examples shown per cluster.

---

#### Q15 — Bridge behaviors (articulation points)

**Must do:**
- Identify nodes whose removal disconnects communities (graph articulation/bridges).

**Pass:**
- Provide list + justification + theological interpretation supported by evidence references.

---

### Category 6: Temporal + Spatial Context

#### Q16 — Dunya vs Akhira mapping

**Must do:**
- Use temporal axis classification rules or evidence.
- Map behaviors to temporal context categories.

**Pass:**
- Behaviors labeled Dunya/Akhira/Both with evidence or explicit "unknown."

---

#### Q17 — Sacred space behaviors

**Must do:**
- Use spatial vocab (Masjid al-Haram, Aqsa, Bayt).
- Return behaviors and expectations tied to locations.

**Pass:**
- Spatial linkage must be evidenced (verse/tafsir mention).

---

### Category 7: Statistics + Patterns

#### Q18 — Surah behavioral fingerprints

**Must do:**
- For each surah: behavior distribution vector.
- Compute similarity between surahs and cluster them.

**Pass:**
- Outputs per-surah distribution computed from deterministic concept index.
- Similarity method stated (cosine, Jaccard, etc.).

---

#### Q19 — Frequency vs centrality discrepancy

**Must do:**
- Correlate mention frequency with centrality.
- Identify undermentioned-high-centrality behaviors.

**Pass:**
- Outputs correlation metric + list of discrepancies + evidence.

---

#### Q20 — Makki/Madani behavioral shift

**Must do:**
- Compute behavior frequency distributions by period.
- Highlight top increases/decreases.

**Pass:**
- Makki/Madani labels must be sourced from a known dataset.
- Show statistical comparison method.

---

### Category 8: Embeddings + Semantics

#### Q21 — 2D semantic landscape of behaviors

**Must do:**
- Use embeddings for behaviors only (controlled).
- Reduce dimensionality (UMAP/TSNE) with fixed seed.

**Pass:**
- Plot/coordinates reproducible (fixed seed).
- Outliers and clusters described with evidence examples.

---

#### Q22 — Contextual meaning drift of الصدق

**Must do:**
- Build context embeddings (per verse or per tafsir chunk occurrences).
- Quantify drift (cluster separation, centroid shifts).

**Pass:**
- Output includes drift metric + examples + evidence anchors.

---

### Category 9: Complex Multi-System

#### Q23 — Complete tawbah analysis (ALL components)

**Must do:**
- Must invoke: concept index + semantic graph + cross-tafsir comparison + stats + (optional embeddings).
- Must provide structured output sections.

**Pass:**
- Deterministic evidence coverage for tawbah across 5 sources.
- Graph relations evidence-backed.
- Debug trace lists all plan steps and sources.

---

#### Q24 — Behavioral prescription generator

**Must do:**
- Input profile → plan a transformation path using semantic graph (not co-occurrence).
- Provide ordered behaviors + expected intermediate states + evidence.

**Pass:**
- Each step has evidence.
- If path is low-confidence, system must say so.

---

#### Q25 — Quranic Behavioral Genome artifact

**Must do:**
- Output a structured artifact with:
  - all behaviors (73)
  - all agents, organs, heart states, consequences
  - all relationships (typed, evidence-backed)
  - provenance for every edge and mapping

**Pass:**
- Artifact is versioned and reproducible.
- Every edge has evidence offsets.
- Node sets match canonical registry exactly.

---

## 3) Implementation Plan for Tests

### 3.1 Create 25 acceptance tests

File: `tests/test_legendary_questions_full.py`

Each test must assert:
- Correct routing: `debug.question_class`
- `debug.fallback_used == False` for structured deterministic parts
- Evidence offsets exist for cited items
- No semantic edges without evidence

### 3.2 Add "failure honesty" tests

- Query an unknown term and ensure the system returns `no_evidence` rather than inventing.

---

## 4) Audit Mode (Recommended)

Add an `audit=true` flag to `/api/proof/query` that:
- Returns raw evidence objects as-is (chunk_id, offsets, quotes, cue phrases)
- Returns deterministic chain outputs in machine-readable form

This is essential for ministry/enterprise review.

---

## 5) Question Class Mapping

| Question | Class | Primary Components |
|----------|-------|-------------------|
| Q1 | `causal_chain` | semantic_graph |
| Q2 | `shortest_path` | semantic_graph |
| Q3 | `reinforcement_loop` | semantic_graph (STRENGTHENS) |
| Q4 | `cross_tafsir_comparative` | concept_index + tafsir_sources |
| Q5 | `makki_madani_analysis` | concept_index + verse_metadata |
| Q6 | `consensus_dispute` | cross_tafsir_comparison |
| Q7 | `behavior_profile_11axis` | concept_index + taxonomy + graph |
| Q8 | `organ_behavior_mapping` | entity_registry + graph |
| Q9 | `state_transition` | semantic_graph (heart_states) |
| Q10 | `agent_attribution` | agent_taxonomy + theological_framing |
| Q11 | `agent_contrast_matrix` | agent_taxonomy + behaviors + consequences |
| Q12 | `prophetic_archetype` | agent_taxonomy + verse_evidence |
| Q13 | `network_centrality` | semantic_graph + metrics |
| Q14 | `community_detection` | semantic_graph + clustering |
| Q15 | `bridge_behaviors` | semantic_graph + articulation |
| Q16 | `temporal_mapping` | temporal_axis + behaviors |
| Q17 | `spatial_mapping` | spatial_vocab + behaviors |
| Q18 | `surah_fingerprints` | concept_index + stats |
| Q19 | `frequency_centrality` | concept_index + graph_metrics |
| Q20 | `makki_madani_shift` | concept_index + verse_metadata |
| Q21 | `semantic_landscape` | embeddings + visualization |
| Q22 | `meaning_drift` | embeddings + context_analysis |
| Q23 | `complete_analysis` | ALL components |
| Q24 | `prescription_generator` | semantic_graph + planner |
| Q25 | `genome_artifact` | ALL components + export |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-01-26 | Initial specification |
