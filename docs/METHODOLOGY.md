# QBM Methodology

## Overview

The Quranic Behavioral Matrix (QBM) methodology is grounded in classical Islamic scholarship, specifically the Bouzidani behavioral classification framework, enhanced with modern NLP and graph-based reasoning.

## Foundational Principles

### 1. Evidence-First Approach

Every behavioral classification must trace back to:
- **Primary Source**: Quranic text (Uthmani script)
- **Classical Commentary**: 5 authoritative tafsir sources
- **Scholarly Consensus**: Validated by the Bouzidani framework

**No Fabrication Rule**: The system never generates claims without provenance. Every assertion includes:
- `chunk_id`: Unique identifier for the evidence source
- `verse_key`: Surah:Ayah reference
- `char_start/char_end`: Exact character offsets in source text
- `quote`: Verbatim text from source

### 2. Deterministic Truth Layer

The truth layer is the foundation of QBM. It consists of:

**Canonical Entities** (`vocab/canonical_entities.json`):
- 73 behaviors (emotional, cognitive, social, physical, spiritual)
- 14 agents (Allah, Believer, Disbeliever, Prophet, etc.)
- 11 organs (heart, tongue, eyes, hands, etc.)
- 12 heart states (alive, dead, sealed, hardened, etc.)
- 16 consequences (Jannah, Jahannam, guidance, misguidance, etc.)

**11-Axis Classification System**:
1. Behavior Form (inner_state, speech_act, relational_act, physical_act, trait)
2. Agent Type (who performs the behavior)
3. Target Type (who/what is affected)
4. Valence (positive, negative, neutral)
5. Intensity (low, medium, high)
6. Temporal Aspect (past, present, future, timeless)
7. Conditionality (conditional, unconditional)
8. Social Scope (individual, interpersonal, communal, universal)
9. Spiritual Domain (worship, ethics, belief, law)
10. Consequence Type (worldly, afterlife, both)
11. Quranic Context (command, prohibition, narrative, parable)

### 3. Semantic Graph Layer

The semantic graph (`data/graph/semantic_graph_v2.json`) captures relationships between entities:

**Edge Types**:
| Type | Meaning | Example |
|------|---------|---------|
| `CAUSES` | Direct causation | Patience → Reward |
| `LEADS_TO` | Sequential progression | Arrogance → Denial → Punishment |
| `PREVENTS` | Protective relationship | Taqwa → Prevents Sin |
| `ASSOCIATED_WITH` | Co-occurrence | Gratitude ↔ Contentment |
| `OPPOSITE_OF` | Antonym | Patience ↔ Impatience |
| `PART_OF` | Hierarchical | Salah ∈ Worship |
| `REQUIRES` | Prerequisite | Hajj requires Ability |

**Edge Properties**:
- `confidence`: 0.0-1.0 score based on evidence strength
- `evidence_count`: Number of supporting verses/chunks
- `provenance`: List of evidence sources

## Tafsir Integration

### Sources

QBM integrates 7 classical tafsir sources:

| Source | Author | Era | Approach |
|--------|--------|-----|----------|
| Ibn Kathir | Ismail ibn Kathir | 14th c. | Hadith-based |
| Tabari | Muhammad ibn Jarir | 10th c. | Comprehensive |
| Qurtubi | Al-Qurtubi | 13th c. | Legal/Fiqh |
| Sa'di | Abdur-Rahman as-Sa'di | 20th c. | Modern Arabic |
| Jalalayn | Jalal ad-Din | 15th c. | Concise |
| Baghawi | Al-Husayn al-Baghawi | 12th c. | Hadith/Athar |
| Muyassar | Ministry of Awqaf | 21st c. | Simplified |

### Chunking Strategy

Tafsir texts are chunked for retrieval:
- **Chunk size**: ~500 characters
- **Overlap**: 50 characters
- **Boundaries**: Respect sentence/paragraph breaks
- **Metadata**: surah, ayah, source, chunk_id

### Cross-Reference Validation

When multiple tafsir sources agree on a behavioral classification:
- Confidence increases
- Evidence is marked as "consensus"
- Disagreements are preserved for scholar review

## Retrieval Methodology

### Query Classification

Queries are classified into types for optimal retrieval:

| Type | Pattern | Strategy |
|------|---------|----------|
| `SURAH_REF` | "سورة البقرة" | Direct lookup |
| `AYAH_REF` | "2:255" | Exact match |
| `CONCEPT_REF` | "الصبر" | Behavior search |
| `FREE_TEXT` | General question | Hybrid retrieval |

### Hybrid Retrieval

1. **BM25 Keyword Search**: Fast, deterministic
2. **FAISS Vector Search**: Semantic similarity
3. **Result Fusion**: Combine and deduplicate
4. **Deterministic Ordering**: Tie-breakers prevent pagination issues

### Evidence Filtering

Before inclusion in proof:
1. Remove placeholder text (e.g., "تفسير غير متوفر")
2. Validate provenance exists
3. Check confidence threshold (default: 0.5)
4. Deduplicate by content hash

## Behavioral Classification Process

### Step 1: Text Extraction

Extract behavioral spans from Quranic text:
- Identify action verbs and state descriptions
- Map to canonical behavior vocabulary
- Record character offsets

### Step 2: Agent Identification

Determine who performs the behavior:
- Parse grammatical subject
- Map to canonical agent types
- Handle implicit agents (e.g., commands to believers)

### Step 3: Axis Classification

Classify along 11 axes:
- Use rule-based heuristics for clear cases
- Use ML models for ambiguous cases
- Flag uncertain classifications for review

### Step 4: Relationship Extraction

Identify relationships between behaviors:
- Parse causal language (فَ، لِ، حَتَّى)
- Identify conditional structures (إِنْ، لَوْ)
- Map to semantic edge types

### Step 5: Validation

Validate classifications:
- Cross-reference with tafsir
- Check consistency with known patterns
- Flag for scholar review if uncertain

## Scholar Review Workflow

### Review Types

| Type | Target | Purpose |
|------|--------|---------|
| `span` | Behavioral span | Validate classification |
| `edge` | Semantic edge | Validate relationship |
| `chunk` | Tafsir chunk | Validate interpretation |

### Review States

```
pending → approved
        → rejected
        → needs_revision
```

### Audit Trail

All review actions are logged:
- Actor ID
- Timestamp
- Old state → New state
- Comments

## Quality Metrics

### Retrieval Quality

| Metric | Target | Description |
|--------|--------|-------------|
| NDCG@10 | 0.70-0.80 | Ranking quality |
| MRR | >0.75 | First relevant result position |
| Recall@k | >0.85 | Coverage of relevant results |

### Classification Quality

| Metric | Target | Description |
|--------|--------|-------------|
| Precision | >0.90 | Correct classifications |
| Recall | >0.85 | Coverage of behaviors |
| F1 | >0.87 | Harmonic mean |

### Bias Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| AraWEAT | <0.15 | Arabic word embedding bias |
| SEAT | <0.15 | Sentence embedding bias |

## Limitations

See `docs/KNOWN_LIMITATIONS.md` for:
- Cases where the system refuses to answer
- Known classification challenges
- Areas requiring human expertise

## References

1. Bouzidani, M. (2020). *Behavioral Classification in the Quran*
2. Ibn Kathir. *Tafsir al-Quran al-Azim*
3. Al-Tabari. *Jami' al-Bayan*
4. Al-Qurtubi. *Al-Jami' li-Ahkam al-Quran*
5. As-Sa'di. *Taysir al-Karim ar-Rahman*
6. Al-Mahalli & As-Suyuti. *Tafsir al-Jalalayn*
