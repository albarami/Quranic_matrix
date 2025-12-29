# QBM Data Architecture & Domain Audit (AS-IS, NO FIXES)

This deliverable is an **audit report only**. It contains **no remediation plan, no recommendations, and no code/data changes**. All statements below are grounded in the current repository state (files under `data/`, `src/`, `scripts/`, `vocab/`, and SQLite/Chroma stores).

---

## A) Executive Summary (1 page max)

### Overall architecture coherence rating

**Not production-grade and not research-grade (prototype / internally inconsistent canonical model).**

The repo contains multiple parallel data representations and pipelines (SQLite, JSON/JSONL, Chroma, NPY indexes, “Full Power” proof), but they do **not** share a single canonical identifier system, a single canonical schema, or a single provenance standard. As a result, a Qur’an/tafsir specialist cannot reliably trace many system outputs back to verifiable verse/tafsir evidence end-to-end.

### Top 10 architectural/semantic risks

1) **End-to-end provenance is not implemented in the data artifacts that drive retrieval/outputs** (no stable span offsets / char offsets / source_file pointers for most artifacts), blocking defensible “show me exactly where this came from” verification.

2) **Multiple incompatible behavior ID systems coexist** (73 vs 46 vs 33 vs 3+ naming schemes), and mappings are partial/inconsistent; this breaks referential integrity across vocab ⇄ annotations ⇄ graph ⇄ vectors.

3) **RAG vector store is structurally empty for verses and tafsir**: Chroma has 73 behavior vectors but **0** ayah vectors and **0** tafsir vectors; the RAG pipeline still queries them.

4) **Proof/graph “evidence” includes explicit synthetic fallbacks** (hard-coded edges/weights/verse counts and heuristic “similarities”), which is not defensible as factual graph truth.

5) **Tafsir text quality is fragmented across stores**: JSONL tafsir is mostly clean, while SQLite tafsir is HTML-heavy and can appear encoding-corrupted in outputs; pipelines mix these representations.

6) **Inconsistent tafsir-source coverage by subsystem**: JSONL has 7 sources with full ayah coverage, but Phase-5 extractor/embeddings cover 4 sources; “Full Power” index covers 5 but is incomplete for some; SQLite DB has 1.

7) **Full Power unified index excludes a Qur’anic subset (muqattaʿāt verses)** from the Quran embedding partition (17 ayat missing), creating silent coverage holes.

8) **Index metadata often lacks stable document IDs**, forcing implicit positional provenance (row index), which breaks reproducibility/traceability after rebuilds and prevents exact citation.

9) **Graph semantics are routinely conflated**: co-occurrence edges are labeled/consumed as “similarity” or “causal” relationships in some layers, and relationship extraction is noisy.

10) **Controlled vocabulary contract is not consistently honored by exported spans** (e.g., agent/systemic/action_class/speech_mode values outside vocab), undermining downstream validation and interpretability for specialists.

---

## B) Current Data Architecture Map (AS-IS)

### Datastores & key artifacts (what exists today)

**1) Qur’an text (canonical token stream)**
- File: `data/quran/uthmani_hafs_v1.tok_v1.json`
- Keys: `surahs[].surah`, `surahs[].ayat[].ayah`, `tokens[].{index,text,start_char,end_char}`
- Identifier: canonical verse key is effectively `(surah, ayah)`; token anchoring exists and is internally consistent (see Section D).

**2) Tafsir (multiple representations)**
- Line-delimited JSONL (multi-source): `data/tafsir/*.ar.jsonl`
  - Record keys: `reference.surah`, `reference.ayah`, `text_ar`, `resource_name`, `tafsir_id`
  - Practical identifier: `(source_from_filename, surah, ayah)`; note `tafsir_id` is not unique per ayah (it is source-level).
- SQLite (single-source, HTML-heavy): `data/tafsir/tafsir.db` (Ibn Kathir only)
  - Table: `tafsir_content(source_id, surah, ayah, text_ar, …)` + FTS tables
- SQLite (single-source, cleaned): `data/tafsir/tafsir_cleaned.db` (Ibn Kathir only)
  - Table: `tafsir_content_cleaned(source_id, surah, ayah, text_original, text_cleaned, html_contamination_rate, …)`
- BM25 “indexes” as JSON: `data/indexes/tafsir/*.json`
  - Shape: `{ \"documents\": [ {surah, ayah, text, source}, ... ] }`
  - Coverage is incomplete for some sources (see Section D).

**3) Span/annotation dataset exports (API-facing)**
- File: `data/exports/qbm_silver_20251221.json`
  - Record keys: `span_id`, `reference.{surah,ayah,surah_name}`, `text_ar`, `behavior_form`, `agent`, `normative`, `action`, `axes`, `evidence`
  - Identifier: `span_id` + `reference.(surah,ayah)`; spans are 1-per-ayah in this export.

**4) SQLite “production” schema (present, but empty data)**
- File: `data/qbm.db`
- Tables: `spans`, `assertions`, `reviews`, `annotators`, `tafsir_consultations`, `coverage_tracker`
- Row counts observed: `spans=0`, `assertions=0`, `coverage_tracker=114` (schema initialized; data not populated).

**5) Graph stores**
- SQLite persisted knowledge graph (vocab-centric): `data/qbm_graph.db`
  - Tables: `nodes(id,node_type,properties)`, `edges(source,target,edge_type,properties)`
  - Observed content: Behavior taxonomy graph only (73 behaviors + 7 categories; no verse/tafsir/spans).
- JSON co-occurrence graph (33 behaviors): `data/behavior_graph_v2.json`
  - `nodes[{id,arabic}]`, `edges[{source,target,weight}]` where `weight` is co-occurrence count.
- JSON relationships from keyword extraction: `data/annotations/tafsir_relationships.json`
  - `causes/effects/opposites/co_mentions` keyed by Arabic tokens (includes non-behavior entities; see Section E/F).
- In-memory unified graph (built at runtime): `src/api/unified_graph.py`
  - Node IDs: `verse:{surah}:{ayah}`, `tafsir:{source}:{surah}:{ayah}`, `behavior:{arabic_keyword}`, `span:{i}` (note span IDs are positional, not `span_id`).

**6) Vector stores / embedding indexes**
- Chroma persistent store: `data/chromadb/` (SQLite metadata + segment files)
  - Collections: `qbm_behaviors`, `qbm_ayat`, `qbm_tafsir`
  - Observed embeddings: behaviors only; ayat/tafsir empty (Section D).
- NPY embedding arrays + metadata:
  - `data/embeddings/annotations_embeddings.npy` + `data/embeddings/annotations_embeddings_metadata.json`
  - `data/embeddings/gpu_index.npy` + `data/embeddings/gpu_index.npy.metadata.json`
  - These are “annotation embedding” indexes (hundreds of thousands of vectors) but metadata lacks text/context for evidence display.
- Full Power mixed-type index:
  - `data/indexes/full_power_index.npy` + `data/indexes/full_power_index.npy.metadata.json`
  - Mixes Quran, tafsir, and behavior-mention snippets into one embedding space.

### System data map diagram (text form) with linkage keys (or missing links)

```
Qur’an Verse
  └─ (surah, ayah) in `data/quran/uthmani_hafs_v1.tok_v1.json`
     └─ tokens have (start_char, end_char) but are NOT referenced by exported spans

Verse → Tafsir Passage
  ├─ JSONL: `data/tafsir/{source}.ar.jsonl` uses `reference.surah` + `reference.ayah`
  └─ SQLite (Ibn Kathir only): `data/tafsir/tafsir*.db` uses `tafsir_content(_cleaned).{source_id,surah,ayah}`

Verse → Span/Annotation (QBM “silver” export)
  └─ `data/exports/qbm_silver_20251221.json`: `spans[].reference.{surah,ayah}` + `spans[].span_id`
     (No token_start/token_end; no char offsets; evidence is not span-grounded)

Span/Annotation → Behavior/Vocabulary
  ├─ Uses mixed conventions:
  │   - `behavior_form` matches `vocab/behavior_form.json` by `items[].en` (not by `id`)
  │   - `normative.{evaluation,deontic_signal,speech_mode}` match vocab by `en` (with exceptions)
  │   - `axes.systemic` uses `SYS_*` but export uses `SYS_SOCIAL` while vocab defines `SYS_SOCIETY`
  │   - `agent.type` includes codes not present in `vocab/agents.json`
  └─ No linkage to the 73-item `vocab/behavior_concepts.json` in the silver export

Behavior/Vocab → Graph
  ├─ `data/qbm_graph.db`: nodes are `BEH_*` IDs from `vocab/behavior_concepts.json`
  ├─ `data/behavior_graph_v2.json`: nodes are a DIFFERENT 33-ID scheme (`BEH_DHIKR`, `BEH_SABR`, …)
  └─ `data/annotations/tafsir_relationships.json`: keys are Arabic surface forms (not `BEH_*`)

Graph → Vectors
  ├─ Chroma (`data/chromadb`): only behavior vectors (73); no verse/tafsir vectors
  ├─ Annotation embeddings (`data/embeddings/*.npy`): vectors keyed by metadata rows (no stable IDs, no text)
  └─ Full Power index (`data/indexes/full_power_index.npy`): vectors keyed by metadata entries (type/source/verse)

Vectors → RAG Answer / Proof Outputs
  ├─ `src/ai/rag/qbm_rag.py`: pulls from Chroma collections (ayat/tafsir empty → weak/empty evidence)
  └─ `src/ml/full_power_system.py` + `src/ml/mandatory_proof_system.py`: uses mixed index + proof scaffolding,
      but includes synthetic fallbacks for graph edges and “embedding similarities”
```

---

## C) Canonical Data Model Assessment

### Does the system have a clear canonical model?

**No single canonical model is consistently enforced across layers.**

Coexisting “canonical candidates” include:
- Tokenized Qur’an schema (strong, consistent): `schemas/quran_tokenized_v1.schema.json` + `data/quran/uthmani_hafs_v1.tok_v1.json`
- Tafsir record schema (minimal): `schemas/tafsir_record_v1.schema.json` + JSONL tafsir files
- SQLAlchemy DB schema for spans/assertions/etc: `src/scripts/setup_database.py` → `data/qbm.db` (but empty)
- API-level span model (simplified): `src/api/models.py` (e.g., `Axes` only models `systemic`)
- Multiple “tafsir annotation” schemas (behavior/relationship/inner_state/speech_act/heart_type):
  - Extraction dataclass: `src/ai/unified/tafsir_extractor.py`
  - Provenance model with offsets (not reflected in stored artifacts): `src/preprocessing/provenance.py`

### Canonical entities observed (and separation quality)

**Verse**
- Well-defined and stable as `(surah, ayah)` across Quran/tafsir/spans/indexes.

**Token**
- Well-defined in tokenized Quran artifact with char offsets.
- Not used by the exported span/annotation records (no token_start/token_end or char offsets).

**TafsirEntry**
- Exists in multiple stores (JSONL multi-source; SQLite single-source; BM25 JSON partial).
- Methodological identity (author/school/style) is not modeled beyond “source” string and optional `resource_name`.

**Span (QBM annotation unit)**
- Exported span is 1-per-ayah and not token/offset-grounded.
- SQL schema supports token anchors and assertions, but the SQLite DB has no span data.

**BehaviorConcept**
- Official controlled vocabulary: 73 concepts (`vocab/behavior_concepts.json`, IDs like `BEH_COG_ARROGANCE`).
- Additional competing behavior-ID schemes appear in training/graph artifacts (33- and 46-behavior subsets and alternate naming conventions).

**Annotation (extracted from tafsir)**
- Stored as JSONL records with `type` + `value` and a context snippet.
- Relationship annotations do not store a stable “from/to” pair; they store relationship type + a captured “target” string in metadata.

### Entity/label conflations (domain-critical)

The “behavior” surface layer in several graph/relationship artifacts mixes:
- **Behaviors** (e.g., صبر)
- **Inner states / emotions** (e.g., غضب)
- **Theological states** (e.g., إيمان/كفر/شرك)
- **Agents/roles** (e.g., نبي/رسول/منافق/كافر)

This conflation appears directly in stored relationship keys and in keyword-based extraction logic, making downstream “behavior graph” semantics methodologically ambiguous for tafsir-aware review.

---

## D) Integrity & Traceability Results (measured where possible)

### D1) Qur’an text integrity (tokenization + offsets)

Artifact checked: `data/quran/uthmani_hafs_v1.tok_v1.json`
- **Verses:** 6,236
- **Tokens:** 81,812
- **Offset validity:** 0 invalid offsets; 0 substring mismatches (token text equals `text[start_char:end_char]`)

Result: the token stream is internally consistent and suitable for span grounding **if** downstream data uses these anchors (most downstream artifacts currently do not).

### D2) Tafsir coverage (per source, per ayah)

**JSONL tafsir (7 sources) — coverage vs tokenized Qur’an**

Each JSONL source is keyed by `reference.surah` + `reference.ayah`.

| Source file | Unique (surah,ayah) | Missing vs 6,236 | Duplicate verse keys |
|---|---:|---:|---:|
| `data/tafsir/ibn_kathir.ar.jsonl` | 6,236 | 0 | 0 |
| `data/tafsir/tabari.ar.jsonl` | 6,236 | 0 | 0 |
| `data/tafsir/qurtubi.ar.jsonl` | 6,236 | 0 | 0 |
| `data/tafsir/saadi.ar.jsonl` | 6,236 | 0 | 0 |
| `data/tafsir/jalalayn.ar.jsonl` | 6,236 | 0 | 0 |
| `data/tafsir/baghawi.ar.jsonl` | 6,236 | 0 | 0 |
| `data/tafsir/muyassar.ar.jsonl` | 6,236 | 0 | **51** (6,287 lines total) |

**SQLite tafsir DBs (single source only)**

| DB | Sources present | Ayah rows | HTML-like rows | Notes |
|---|---|---:|---:|---|
| `data/tafsir/tafsir.db` | `ibn_kathir` only | 6,236 | 6,198 | `text_ar` is predominantly HTML-wrapped |
| `data/tafsir/tafsir_cleaned.db` | `ibn_kathir` only | 6,236 | `text_cleaned`: 0 | Avg `html_contamination_rate` ≈ 0.318 |

Result: multi-source tafsir is present in JSONL, but SQLite tafsir storage is single-source and HTML-heavy.

### D3) Span/export coverage (API-facing)

Artifact checked: `data/exports/qbm_silver_20251221.json`
- **Spans:** 6,236
- **Unique span_id:** 6,236 (no duplicates)
- **Unique verses referenced:** 6,236 (exactly 1 span per ayah)
- **Axes present in every span:** only `situational` (internal/external) and `systemic` (SYS_GOD/SYS_SOCIAL)

**Controlled vocabulary integrity (measured mismatches)**

Measured against vocab files under `vocab/` (using the repo’s current vocab definitions):
- `agent.type` not in `vocab/agents.json`: **2,873 spans affected** (`AGT_ALLAH`: 2,855; `AGT_PROPHET`: 18)
- `normative.speech_mode` not in `vocab/speech_mode.json` (`en` values): **1,527 spans affected** (`interrogative`)
- `action.class` not in `vocab/action_class.json`: **3,338 spans affected** (`ACT_PSYCHOLOGICAL`)
- `axes.systemic` not in `vocab/systemic.json`: **3,686 spans affected** (`SYS_SOCIAL` vs vocab’s `SYS_SOCIETY`)

Result: the export is verse-complete, but not referentially consistent with the controlled vocab contract for multiple key dimensions.

### D4) Behavior taxonomy and “73 vs 46 vs 33” reconciliation (measured)

**73 (official vocabulary)**
- File: `vocab/behavior_concepts.json`
- Count: **73** behavior concepts (IDs like `BEH_SPEECH_TRUTHFULNESS`, `BEH_COG_ARROGANCE`, …)

**46 (keyword-extracted tafsir behavioral mentions)**
- File: `data/annotations/tafsir_behavioral_annotations.jsonl`
- Lines: **68,240**
- Unique `behavior_ar`/`behavior_en`: **46**
- No `behavior_id` field (IDs are Arabic surface forms)

**33 (ML/graph subset across multiple ID schemes)**
- Files:
  - `data/annotations/tafsir_behavioral_5axis.jsonl` → 33 IDs like `BEH_DHIKR`, `BEH_SABR`, …
  - `data/annotations/tafsir_behavioral_annotations_filtered.jsonl` → 33 IDs like `BEH_CHAR_OPPRESSION`, `BEH_HEART_FAITH`, `BEH_WORSHIP_REMEMBRANCE`, …
  - `data/annotations/tafsir_behavioral_annotations_schema.jsonl` → 33 IDs like `BEH_SOC_OPPRESSION`, `BEH_COG_ARROGANCE`, …
  - `data/behavior_graph_v2.json` → **33** nodes (same short-ID family as `BEH_DHIKR`, …)

**Mismatch inside the “schema” subset**
- In `data/annotations/tafsir_behavioral_annotations_schema.jsonl`, **9,113 / 47,142** records use `behavior_id` values that do **not** exist in `vocab/behavior_concepts.json` (≈19.3%).
- Missing-from-vocab IDs and their record counts:
  - `BEH_SPI_SHIRK`: 4,463
  - `BEH_SOC_MERCY`: 1,359
  - `BEH_EMO_CONTENTMENT`: 1,080
  - `BEH_SOC_EXCELLENCE`: 825
  - `BEH_PHY_MODESTY`: 566
  - `BEH_SOC_TRANSGRESSION`: 340
  - `BEH_FIN_ASCETICISM`: 123
  - `BEH_SPI_KHUSHU`: 109
  - `BEH_SOC_TRUSTWORTHINESS`: 138
  - `BEH_SOC_BETRAYAL`: 110

Result: the repo contains at least **four** behavior identity layers (73 / 46 / 33-short / 33-schema), and even the “schema” file is not fully aligned to the official 73-item vocabulary.

### D5) Graph integrity (IDs, edge semantics, orphans)

**Persisted graph (`data/qbm_graph.db`)**
- Nodes: **80** (73 `Behavior` + 7 `BehaviorCategory`)
- Edges: **111** (`RELATED`: 73, `OPPOSITE_OF`: 20, `SIMILAR_TO`: 8, `CAUSES`: 10)
- Orphan edges (edge endpoints missing as nodes): **0**

Result: the persisted graph is internally consistent as a **vocabulary graph**, but it is not a verse/tafsir-evidence graph.

### D6) Vector stores and traceability (measured)

**Chroma (`data/chromadb`)**
- Collections exist: `qbm_ayat`, `qbm_behaviors`, `qbm_tafsir`
- Embeddings present:
  - `qbm_behaviors`: **73**
  - `qbm_ayat`: **0**
  - `qbm_tafsir`: **0**

Result: the RAG pipeline that queries ayat and tafsir vectors cannot retrieve them from this persisted store as currently populated.

**Annotation embedding index (`data/embeddings/annotations_embeddings.npy`)**
- Shape: **(322,939, 768)**; metadata list length: **322,939**
- Sources in metadata: **4** (`ibn_kathir`, `tabari`, `qurtubi`, `saadi`)
- Metadata fields: `{surah, ayah, source, type, value}` (no stored context text; no stable IDs)

**Mismatch vs current raw annotation file**
- Raw annotations file: `data/annotations/tafsir_annotations.jsonl` has **309,411** lines (same 4 sources).
- Comparing tuples `(surah,ayah,source,type,value)`:
  - Embedding-metadata rows whose tuple exists anywhere in current raw file: **306,267**
  - Embedding-metadata rows with no matching tuple in current raw file: **16,672**

Result: a non-trivial slice of vector items cannot be traced back to any current raw record, even at the tuple level (without considering context).

### D7) BM25 tafsir index coverage (measured)

Artifacts checked: `data/indexes/tafsir/*.json` (7 sources)
- Total documents across sources: **38,897** (expected 43,652 for 7×6,236 if complete)

| Source | Docs | Missing vs 6,236 | Duplicates |
|---|---:|---:|---:|
| ibn_kathir | 6,202 | 34 | 0 |
| tabari | 6,235 | 1 | 0 |
| qurtubi | 6,193 | 43 | 0 |
| saadi | 6,225 | 11 | 0 |
| jalalayn | 6,191 | 45 | 0 |
| muyassar | 5,613 | 1,055 | 432 |
| baghawi | 2,238 | 3,998 | 0 |

Result: any system component relying on these BM25 JSON indexes does not have uniform multi-source tafsir coverage.

### D8) Full Power mixed index coverage (measured)

Artifact checked: `data/indexes/full_power_index.npy.metadata.json` (113,865 items)
- Type partition:
  - `quran`: 6,219 (missing 17 verses)
  - `tafsir`: 31,066 (missing 114 tafsir entries vs 5×6,236)
  - `behavior`: 76,580
- Qur’an verses missing from the `quran` partition are the 17 muqattaʿāt-only ayat:
  - `2:1`, `3:1`, `20:1`, `26:1`, `28:1`, `29:1`, `30:1`, `31:1`, `32:1`, `36:1`, `40:1`, `41:1`, `42:1`, `43:1`, `44:1`, `45:1`, `46:1`
- Tafsir per-source counts in the index (5 sources only):
  - `tabari`: 6,236 (complete)
  - `ibn_kathir`: 6,221 (missing 15)
  - `saadi`: 6,225 (missing 11)
  - `qurtubi`: 6,193 (missing 43)
  - `jalalayn`: 6,191 (missing 45)
- Tafsir text contamination inside this index:
  - **17,930 / 31,066** tafsir items contain HTML-like markup (`<...>`)
- Metadata has **no stable `id` field**; uniqueness is implicit by `(type, source, verse[, behavior])`.

Result: the “Full Power” index is not verse-complete for Quran, not tafsir-complete for all 5 sources, and includes significant HTML markup in embedded text.

---

## E) Arabic/Qur’anic Domain Validity Review

### E1) Can a Qur’an/tafsir specialist trace outputs back to evidence?

**Verse-level traceability exists** wherever outputs cite `(surah, ayah)`—this key is stable across Qur’an tokenization, JSONL tafsir, and exported spans.

However, **evidence-level traceability is systematically weak** because:
- Exported spans do not carry token/char offsets, despite the Qur’an token stream supporting them.
- Extracted tafsir annotations store context snippets but do not store offsets, and vector metadata often drops the context text entirely.
- Some proof outputs show corrupted tafsir text encoding and/or HTML, undermining scholarly readability and verification.

### E2) Tafsir methodological differences (Tabari vs Qurtubi vs Jalalayn…)

The data model does not preserve tafsir methodological distinctions beyond a source string:
- No explicit representation of tafsir type (riwāyah-heavy like al-Ṭabarī vs fiqh/legal focus like al-Qurṭubī vs concise paraphrase like al-Jalalayn).
- Retrieval pipelines often treat sources as interchangeable “documents” with optional source-diversity heuristics.

This flattens scholarly differences at the data model level, making it hard to defend methodology-sensitive claims (e.g., legal inference vs narrative transmission vs linguistic gloss).

### E3) Behavior vs state vs agent conflation (domain correctness)

Multiple artifacts treat theological states and roles as “behaviors” in graph/relationship contexts:
- `data/annotations/tafsir_relationships.json` keys include roles/entities such as **نبي / رسول / منافق / كافر** alongside behavior-like terms.
- Keyword-based extractors and “unified graph” logic use surface tokens as behavior nodes (e.g., `behavior:{arabic_keyword}`), not controlled taxonomy IDs.

For tafsir-aware evaluation, this is methodologically risky: “مؤمن/كافر” are often **identity/وصف** categories, not behaviors, and “نبي/رسول” are roles; treating them as behavioral nodes can create invalid causal/semantic edges.

### E4) Muqattaʿāt handling

The Full Power index’s Quran partition excludes the 17 muqattaʿāt-only verses. These verses have domain significance (tafsir debates, rhetorical structure) and their absence biases retrieval and any downstream “coverage” claims made from that index.

### E5) Theological framing risk: “Allah as agent”

The exported span dataset uses `AGT_ALLAH` as an agent code at scale, while the controlled agent vocabulary in `vocab/agents.json` does not define it. Beyond a technical mismatch, this creates a domain framing risk because “agency” attribution lacks explicit definitional boundaries in the data model, allowing category mistakes (divine speech acts vs human actions vs narrative voice).

---

## F) Findings by Severity (NO FIXES)

Each finding includes: what is happening, why it is a problem (technical + domain), where it appears, and evidence.

### Critical

**C1. Provenance/grounding is not end-to-end (no offsets; no stable evidence pointers)**
- What is happening: Most artifacts that drive retrieval/outputs do not include token/char offsets or stable pointers to the exact source span. A provenance schema exists but is not reflected in stored data.
- Why it is a problem:
  - Technical: cannot perform deterministic “show me the exact span” verification; cannot reproduce evidence chunks reliably.
  - Domain: tafsir-aware review requires exact grounding; without offsets, meaning can be altered by truncation or noisy extraction.
- Where: `data/exports/qbm_silver_20251221.json` (no token/char offsets), `data/annotations/*.jsonl` (no offsets), `src/preprocessing/provenance.py` (expects offsets but data doesn’t provide them).
- Evidence: Qur’an tokenization provides offsets (Section D1), but exported spans do not reference them; extracted tafsir annotations store only snippet strings.

**C2. RAG vector store has no ayah/tafsir vectors, but RAG queries them**
- What is happening: `data/chromadb` has 73 behavior embeddings and 0 ayah/tafsir embeddings, while `src/ai/rag/qbm_rag.py` queries `search_ayat()` and `search_tafsir()`.
- Why it is a problem:
  - Technical: RAG retrieval for verses/tafsir returns empty/weak evidence; outputs cannot be evidence-backed.
  - Domain: answers can appear scholarly while lacking verifiable Qur’an/tafsir citations.
- Where: `data/chromadb/chroma.sqlite3` (counts), `src/ai/rag/qbm_rag.py`, `src/ai/vectors/qbm_vectors.py`.
- Evidence: Chroma counts: `qbm_ayat=0`, `qbm_tafsir=0`, `qbm_behaviors=73` (Section D6).

**C3. Synthetic “graph evidence” and heuristic “embedding evidence” are explicitly generated in the proof system**
- What is happening: The proof pipeline can fabricate graph edges/weights and similarity evidence when primary retrieval returns no nodes.
- Why it is a problem:
  - Technical: output is not derived from persisted graph/index state; not reproducible or auditable.
  - Domain: presenting fabricated edges/weights/verse counts as “evidence” is methodologically indefensible for scholarly use.
- Where: `src/ml/mandatory_proof_system.py`
- Evidence:
  - Fallback edges are hard-coded with weights/verse counts (e.g., opposite/leads_to/strengthens edges).
  - Placeholder edge generation occurs even when a graph exists (`{\"from\": \"سلوك1\", ...}` patterns).

**C4. Behavior identity fragmentation (73 vs 46 vs 33, multiple naming schemes) breaks referential integrity**
- What is happening: Different subsystems use different behavior identity spaces (official 73 IDs, 46 Arabic keywords, 33 short IDs, 33 schema IDs, BEH_CHAR/BEH_HEART/BEH_WORSHIP variants).
- Why it is a problem:
  - Technical: prevents stable joins across vocab ⇄ graph ⇄ vectors ⇄ training data; increases “silent mismatch” risk.
  - Domain: a specialist cannot know which taxonomy a label belongs to, or whether two “behaviors” are intended to be the same construct.
- Where: `vocab/behavior_concepts.json`, `data/annotations/tafsir_behavioral_annotations.jsonl`, `data/behavior_graph_v2.json`, `data/annotations/tafsir_behavioral_annotations_schema.jsonl`, `data/training_splits/*.jsonl`, `src/ml/qbm_vocab_mapping.py`.
- Evidence: counts and mismatches in Section D4; 9,113 “schema” records use IDs absent from the official vocab.

**C5. Tafsir text appears encoding-corrupted and HTML-wrapped in proof outputs**
- What is happening: Proof outputs can include mojibake (UTF-8 decoded as Latin-1-like) and HTML markup in tafsir excerpts.
- Why it is a problem:
  - Technical: downstream display/search/reranking can be distorted; text processing may treat markup as content.
  - Domain: scholars cannot reliably read/verify the Arabic; meaning can be corrupted.
- Where: SQLite tafsir `data/tafsir/tafsir.db` (`text_ar` HTML-heavy); proof output sample `test_response.json`.
- Evidence: `test_response.json` contains tafsir snippets like `ÙÙÙÙÙ` with `<p><span…>` markup; Quran evidence list is empty in that sample.

**C6. Vector index traceability is broken across time for extracted tafsir annotations**
- What is happening: The annotation embeddings index (322,939 vectors) does not fully match the current extracted annotation file (309,411 lines) even at the tuple level.
- Why it is a problem:
  - Technical: cannot deterministically trace a vector hit back to a current raw record; index may be stale relative to source data.
  - Domain: evidence cannot be audited (which tafsir text span produced this vector?).
- Where: `data/embeddings/annotations_embeddings.npy`, `data/embeddings/annotations_embeddings_metadata.json`, `data/annotations/tafsir_annotations.jsonl`
- Evidence: 16,672 embedding-metadata rows have no matching `(surah,ayah,source,type,value)` tuple in the current raw file (Section D6).

### High

**H1. “Full Power” Quran partition silently drops 17 muqattaʿāt verses**
- What is happening: The Full Power mixed index includes only 6,219 Quran items, missing 17 verses (all muqattaʿāt-only ayat).
- Why it is a problem:
  - Technical: retrieval coverage claims from this index are inaccurate; downstream analyses are biased.
  - Domain: muqattaʿāt are tafsir-significant; excluding them is not methodologically neutral.
- Where: `data/indexes/full_power_index.npy.metadata.json`
- Evidence: missing list in Section D8.

**H2. Multi-source tafsir coverage is inconsistent across retrieval substrates**
- What is happening: JSONL tafsir is complete for 7 sources, but BM25 JSON indexes are incomplete for some sources (especially baghawi/muyassar), and SQLite DBs are single-source.
- Why it is a problem:
  - Technical: “guaranteed all-sources” retrieval cannot be defended if the chosen substrate is incomplete.
  - Domain: cross-tafsir comparisons become selectively incomplete, biasing interpretation.
- Where: `data/tafsir/*.ar.jsonl`, `data/indexes/tafsir/*.json`, `data/tafsir/tafsir*.db`, `src/ml/stratified_retriever.py`
- Evidence: BM25 counts show baghawi missing 3,998 ayat; muyassar missing 1,055 and duplicates (Section D7).

**H3. Exported spans violate controlled vocab contract for key dimensions**
- What is happening: The verse-complete export uses codes not present in vocab (agent/systemic/action_class/speech_mode).
- Why it is a problem:
  - Technical: validation and cross-layer joins fail; UI filters and analytics can miscount or misclassify.
  - Domain: categories like agent/systemic context are interpretive; inconsistencies are hard to defend.
- Where: `data/exports/qbm_silver_20251221.json`, `vocab/*.json`
- Evidence: mismatch counts in Section D3.

**H4. Relationship extraction produces noisy targets and collapses relationship meaning**
- What is happening: Relationship annotations store `value` as only the relation type (CAUSES/RESULTS_IN/OPPOSITE_OF) and capture a “target” token that can be punctuation or non-behavior text.
- Why it is a problem:
  - Technical: cannot reconstruct a graph edge as (from,to,type) deterministically; relationship vectors are not meaningful.
  - Domain: causal claims in tafsir require high precision; noisy extraction undermines scholarly trust.
- Where: `data/annotations/tafsir_annotations.jsonl` and embedding metadata; extractor logic in `src/ai/unified/tafsir_extractor.py`
- Evidence: relationship duplicates exist only because “to” is not modeled in the `value`; 173 targets contain no Arabic letters (Section D6).

### Medium

**M1. `tafsir_id` is not a per-ayah identifier in JSONL tafsir**
- What is happening: JSONL tafsir records commonly use a source-level `tafsir_id` (e.g., `ibn_kathir`) rather than a unique record ID.
- Why it is a problem:
  - Technical: consumers must rely on composite keys and file context; IDs are not self-contained.
  - Domain: citation systems benefit from explicit, stable record identifiers.
- Where: `data/tafsir/*.ar.jsonl`, `schemas/tafsir_record_v1.schema.json`
- Evidence: `tafsir_id` observed as constant per file; uniqueness depends on `(source, surah, ayah)`.

**M2. In-memory unified graphs use positional span IDs, not `span_id`**
- What is happening: `src/api/unified_graph.py` builds span nodes as `span:{i}` based on list order, not `span_id`.
- Why it is a problem:
  - Technical: graph node IDs are unstable across data reloads/exports; cannot cite a span deterministically.
  - Domain: undermines scholarly audit of “which annotation is this?”
- Where: `src/api/unified_graph.py`
- Evidence: span node creation uses `span_id = f\"span:{i}\"`.

**M3. Multiple tafsir-source lists are hard-coded differently across modules**
- What is happening: some modules assume 5 sources, others 7, others 4.
- Why it is a problem:
  - Technical: coverage and guarantees differ per subsystem; cross-component comparisons are inconsistent.
  - Domain: “all sources” claims become ambiguous.
- Where: `src/api/unified_brain.py` (5), `src/api/tafsir_integration.py` (5), `src/ai/tafsir/cross_tafsir.py` (7), `src/ai/unified/tafsir_extractor.py` (4), extracted artifacts in `data/annotations` (4/5/7).

### Low

**L1. Environment file contains live credentials in-repo**
- What is happening: `.env` contains API keys/secrets.
- Why it is a problem:
  - Technical: operational/security risk (not directly part of data semantics, but affects system governance).
- Where: `.env`
- Evidence: keys present (values not reproduced in this report).

---

## G) Systemic Risk Summary

### If used by researchers
- High risk of **non-reproducible results** due to index/data drift and missing stable IDs.
- High risk of **methodological critique**: taxonomy fragmentation, conflation of behaviors/agents/states, and lack of tafsir-method metadata undermine defensible claims.
- Evidence trails are incomplete (no offsets; vector metadata missing text), preventing rigorous peer verification.

### If used by public users
- High risk of **overconfident unsourced answers**: RAG subsystems can return answers without verse-level Quran evidence and with corrupted tafsir excerpts.
- High risk of **misinterpretation** if co-occurrence/heuristics are presented as causal theological relationships.

### If used by a ministry / Awqaf institution
- High reputational and governance risk:
  - Outputs may not be auditable to primary evidence.
  - Mixed and sometimes corrupted tafsir text representations can cause misquotation.
  - Synthetic “evidence” fallbacks (graph edges/weights) are not institutionally defensible.
