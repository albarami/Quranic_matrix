# Quranic Behavioral Matrix (QBM): What This Codebase Builds

## Abstract

This repository is a full **data + tooling + API + UI** stack for building and exploring a *Quran-wide behavioral classification dataset*.

At its core is the **Quranic Behavioral Matrix (QBM)**: a structured dataset that assigns behavior-focused annotations across **all 6,236 ayat** of the Quran (100% coverage), grounded in Islamic scholarship and in particular **Dr. Ibrahim Bouzidani's five-context behavioral framework**. Around that dataset, the repo contains:

- a reproducible **annotation pipeline** (Label Studio interface, converters, validation, QA, tiered exports)
- a production-ready **FastAPI backend** to query the dataset and tafsir
- a modern **Next.js research frontend** with **generative UI** chat (C1/Thesys)
- an "AI system" layer (knowledge graph, vector retrieval, ontology, multi-tafsir extraction, GPU search, discovery endpoints)

The result is not just a dataset dump - it's an end-to-end research platform for asking questions like:

- "How does the Quran frame *patience* - as inner state, speech, or action?"
- "Which agent-types are most associated with warnings vs praise?"
- "What do multiple tafsir sources emphasize or agree on for a verse?"
- "Which ayat are semantically similar or behaviorally linked?"

---

## 1) The Big Idea: Behavioral Semantics at Quran Scale

Most Quranic studies tools are verse browsers or translation/tafsir readers. QBM aims at a different axis: **human behavior as a structured, queryable domain**.

Instead of searching raw text and manually interpreting results, QBM turns the entire Quran into a **behavioral index** with:

- consistent categories and controlled vocabularies
- traceable annotation choices
- measurable inter-annotator reliability (IAA)
- repeatable exports for downstream analytics or machine learning

The codebase embodies a "research pipeline mindset": build data carefully, validate and measure it, then expose it through interfaces that make scholarship faster - not looser.

---

## 2) The Framework: Five Contexts + Extended Dimensions

The repository centers on a behavioral classification framework with **five contexts** (the Bouzidani foundation) and a broader "dimensional analysis" layer used by the API.

### The 5 Bouzidani contexts (core)
These are treated as axes/contexts in the dataset and documentation:

1. **Organic**: which organs are implicated (e.g., heart, tongue, eyes, hands)
2. **Situational**: internal vs external (and behavior form granularity)
3. **Systemic**: social system / domain of life (e.g., SYS_GOD vs SYS_SOCIAL)
4. **Spatial**: where the behavior occurs (location contexts)
5. **Temporal**: when it occurs (time contexts)

### The 11 "mandatory dimensions" (analysis layer)
In `src/api/dimensions.py`, the project formalizes a comprehensive analysis approach that expands beyond the 5 contexts (e.g., agent, evaluation, consequence, relationships) and provides a methodology prompt for "never skip dimensions" analysis.

This is what powers the richer analytical endpoints in the API.

---

## 3) The Dataset: A Span Record per Ayah (Silver Export)

The repo ships tiered exports under `data/exports/`. The latest silver export (`data/exports/qbm_silver_20251221.json`) is:

- **6,236 spans** (one per ayah in this export)
- includes Arabic verse text (with tashkeel), reference metadata, and annotations

Example fields you see in a span record:

- `reference`: `{surah, ayah, surah_name}`
- `text_ar`: the ayah text
- `behavior_form`: e.g., `inner_state`, `speech_act`, `physical_act`, `relational_act`, `trait_disposition`
- `agent`: `{type, explicit, referent?}`
- `normative`: `{speech_mode, evaluation, deontic_signal}`
- `action`: `{class, textual_eval}`
- `axes`: `{situational, systemic, ...}` (with room for the other contexts)

### Real dataset distributions (from the silver export)

Across the 6,236 silver spans:

- **Agent types** (top):  
  - `AGT_ALLAH`: 2,855  
  - `AGT_DISBELIEVER`: 1,142  
  - `AGT_BELIEVER`: 1,111  
  - `AGT_HUMAN_GENERAL`: 1,082  
- **Behavior forms**:  
  - `inner_state`: 3,184  
  - `speech_act`: 1,255  
  - `relational_act`: 1,003  
  - `physical_act`: 640  
  - `trait_disposition`: 154  
- **Evaluation**: `neutral` dominates (4,808), with `blame` (843) and `praise` (585)
- **Deontic signals**: mostly `khabar` (4,741), with `amr` (444), `tarhib` (639), `targhib` (402), and rare `nahy` (10)
- **Systemic axis**: `SYS_SOCIAL` (3,686) vs `SYS_GOD` (2,550)
- **Situational axis**: `internal` (4,341) vs `external` (1,895)

These numbers are important because they show the project is not theoretical: the dataset is already consistent enough to summarize, chart, and serve to a UI.

---

## 4) Controlled Vocabularies: Freezing Meaning into IDs

A defining design choice here is the use of **controlled vocabularies** under `vocab/`:

- agents (`AGT_*`)
- organs (`ORG_*`) + semantic domains for "heart"
- systemic/spatial/temporal contexts (`SYS_*`, `LOC_*`, `TMP_*`)
- deontic signals, evaluation labels, speech modes, etc.
- behavior concepts (`BEH_*`) organized by categories (speech, emotional, spiritual, social, cognitive, etc.)

This prevents "annotation drift" (where labels mutate over time), and allows:

- strict validation
- reliable export formats
- easier downstream ML / analytics

Validation utilities exist explicitly to enforce this (`src/validation/validate_vocabularies.py`).

---

## 5) Annotation Pipeline: From Quran Text -> Tasks -> Exports

This codebase doesn't assume annotation happens magically. It contains the mechanics to create, run, validate, and iterate an annotation project.

### 5.1 Tokenization contract and pilots

The repo includes schemas and tooling to ensure Quran text and token offsets are stable:

- `schemas/quran_tokenized_v1.schema.json`
- pilot tools under `tools/pilot/` to build selections and tasks

The key idea: **span selection must be convertible to token indices**, which is why tasks include token objects with `start_char`/`end_char`.

### 5.2 Label Studio configuration (human annotation UX)

In `label_studio/` you have:

- `labeling_interface.xml` (a full UI schema)
- sample/pilot task formats
- a documented workflow to load tasks and export annotations

The interface is aligned to the controlled vocabulary IDs and supports:

- span boundary selection
- agent + behavior form
- five-context axes
- normative classification (speech mode, evaluation, deontic)
- behavior concept taxonomy selection

### 5.3 Export conversion: fail-closed mapping into QBM records

Label Studio exports are not your final data model. The converter in `tools/label_studio/convert_export.py` converts UI outputs into QBM records:

- maps UI strings -> frozen IDs
- converts character spans -> token spans using task tokens
- fails closed on unknown values (to protect dataset integrity)

There's explicit test coverage for this converter (`tests/tools/label_studio/test_convert_export.py`).

### 5.4 Quality checks, coverage auditing, and tiered exports

The pipeline includes scripts to keep the dataset scientifically "tight":

- coverage audits (`src/scripts/coverage_audit.py`) -> confirm 114/114 surahs and 6,236/6,236 ayat
- progress reporting (`src/scripts/progress_report.py`)
- automated quality checks (`src/scripts/quality_check.py`)
- tiered export generation (`src/scripts/export_tiers.py`) -> gold/silver/research

### 5.5 Reliability measurement (IAA)

IAA is treated as a first-class research requirement:

- `iaa_results.json` shows micro-pilot reliability with **avg Cohen's kappa ~= 0.925**
- scripts exist for computing field-level agreement and handling low-variance fields

This is crucial: it means the dataset isn't just "annotated"; it's *measured*.

---

## 6) Tafsir: Bringing Scholarship into the Loop

QBM uses tafsir in two ways:

1. **as an annotation aid** (when verse meaning/agent/behavior is ambiguous)
2. **as a parallel corpus for extraction and discovery** (Phase 4+)

### 6.1 Download and local storage

`tools/tafsir/download_tafsir.py` downloads tafsir from Quran.com API and stores it as JSONL.

In `data/tafsir/` the repo contains four complete sources (6,236 ayat each):

- Ibn Kathir (~19.0 MB)
- Tabari (~45.7 MB)
- Qurtubi (~19.5 MB)
- Saadi (~7.4 MB)

Total: **24,944 tafsir records** (~91.6 MB).

### 6.2 SQLite tafsir database + lookup tooling

For fast local workflow, the project builds a SQLite DB with FTS (`tools/tafsir/setup_tafsir_db.py`) and includes a lookup/search tool (`tools/tafsir/tafsir_lookup.py`).

This directly supports the "tafsir consultation protocol" in `docs/training/TAFSIR_CONSULTATION_PROTOCOL.md`.

### 6.3 Cross-tafsir comparison and consensus

`src/ai/tafsir/cross_tafsir.py` implements a multi-source tafsir analyzer that can:

- retrieve tafsir per ayah per source
- compare interpretations and text lengths
- search for behavioral terms in tafsir
- compute "consensus" style signals (what multiple sources all mention)

---

## 7) Backend: FastAPI QBM REST API

The backend in `src/api/main.py` turns the dataset into a real service.

### Core dataset endpoints

- `GET /datasets/{tier}`: full tier export (gold/silver/research)
- `GET /spans`: filter/search spans
- `GET /spans/{id}`: fetch a span by ID
- `GET /surahs` and `GET /surahs/{num}`: surah summaries and verse spans
- `GET /stats`: dataset statistics used by the frontend
- `GET /vocabularies`: controlled vocabulary surfaces for UI filters

The API caches dataset loads and supports basic filtering (surah, agent, evaluation, etc.).

### Tafsir endpoints (frontend integration)

- `GET /tafsir/{surah}/{ayah}`: tafsir for an ayah from selected sources
- `GET /tafsir/compare/{surah}/{ayah}`: compare tafsir across available sources
- `GET /ayah/{surah}/{ayah}`: return the ayah + its QBM annotation summary
- `POST /api/spans/search`: a POST-friendly search endpoint used by C1 tools, including diacritics-stripped matching

### Dimensional analysis endpoints (research features)

The API also includes "analysis-as-a-service" endpoints intended for the research UI:

- `GET /api/analyze/behavior`: analyze a search term across multiple dimensions
- `GET /api/analyze/agent/{agent_type}`: behavior profiles per agent type
- `POST /api/analyze/comprehensive`: multi-dimension analysis guided by the dimensional thinking framework
- `GET /api/behavior/{behavior_name}/map`: generate an 11-dimension "behavior map"

### Discovery routes

`src/api/discovery_routes.py` exposes GPU-powered discovery capabilities under `/discovery/*` (semantic search, cross references, clustering, networks). The design is "lazy-loaded" to avoid GPU initialization at import time.

---

## 8) Frontend: QBM Research Platform (Next.js + Generative UI)

The `qbm-frontend/` app is a research UI built with:

- Next.js 14
- Tailwind + Framer Motion
- Thesys/C1 generative UI SDK

Key pages include:

- `/research`: a chat-first research assistant (C1Chat)
- `/dashboard`: auto-generated charts and summary sections via prompts
- `/explorer` and `/insights`: dataset exploration and pattern discovery views

The frontend's `/api/chat` route streams responses from Thesys' embed endpoint and (when available) pulls live stats from the FastAPI backend (`QBM_BACKEND_URL`).

This is a deliberate UX choice: the interface is not a fixed dashboard - it's a "research console" where the UI itself can be generated dynamically in response to the question asked.

---

## 9) The QBM AI System: From Dataset -> Retrieval -> Discovery

Under `src/ai/`, the repo implements a modular AI layer that is meant to sit on top of QBM data.

### 9.1 Knowledge graph (NetworkX + SQLite)

`src/ai/graph/qbm_graph.py` implements a persisted knowledge graph:

- **73 behaviors**
- **80 total nodes**
- **111 edges** across relationship types (CAUSES, OPPOSITE_OF, SIMILAR_TO, RELATED)

It supports queries like:

- causal chain finding
- hub behavior detection
- community discovery

### 9.2 Vector store (ChromaDB + Arabic embeddings)

`src/ai/vectors/qbm_vectors.py` provides persistent semantic search collections for:

- ayat
- behaviors
- tafsir

It uses Arabic-optimized embedding models (AraBERT preferred), with fallback behavior for test/dev environments.

### 9.3 RAG pipeline (vector retrieval + graph expansion + Azure OpenAI)

`src/ai/rag/qbm_rag.py` combines:

- vector retrieval (ayat/behavior/tafsir similarity)
- knowledge-graph expansion (causes/effects/opposites)
- Azure OpenAI generation (configured by env vars)

This is meant for scholar-facing Q&A that still stays grounded in retrieved evidence.

### 9.4 Ontology layer (RDFLib, OWL, SPARQL)

`src/ai/ontology/qbm_ontology.py` builds an OWL ontology for QBM concepts and supports SPARQL queries like causal-chain traversal.

This enables semantic reasoning and interoperability with "linked data" workflows.

### 9.5 Unified system (single interface across components)

`src/ai/unified/qbm_unified.py` provides an integrated query interface linking:

- multi-tafsir retrieval
- behavior taxonomy and roots
- knowledge graph relationships
- vector search primitives

It supports "Ayah queries", "Behavior queries", "Concept queries", and "Consensus queries".

### 9.6 Phase 5: Tafsir annotation extraction (322,939 annotations)

`src/ai/unified/tafsir_extractor.py` is a major scale step: it automatically extracts structured annotations from tafsir text across all 4 sources.

From `data/annotations/tafsir_annotations.jsonl`:

- **322,939 total extracted annotations**
  - behaviors: 256,962
  - speech acts: 45,625
  - inner states: 13,666
  - relationships: 6,669
  - heart types: 17

This turns tafsir into a second, massive annotation layer - useful for discovery, retrieval, and evidence-backed analysis.

---

## 10) Phase 6: GPU Acceleration (Embeddings, Search, Reranking)

The project includes serious GPU tooling under `src/ai/gpu/`:

- multi-GPU embedding generation with AraBERT (`gpu_embeddings.py`)
- Windows-compatible PyTorch GPU vector search (`gpu_search.py`)
- cross-encoder reranking for precision (`reranker.py`)

Helper scripts in `scripts/` orchestrate the phases:

- `scripts/run_phase5_extraction.py`
- `scripts/run_phase6_gpu.py`
- `scripts/run_phase7_discovery.py`

This is what makes "search across 322k annotations" interactive rather than academic.

---

## 11) Testing: Research Code with Guardrails

The repository has a real test suite (`tests/`) covering:

- the FastAPI API contract (`tests/test_api.py`)
- AI modules (graph, vectors, tafsir, unified, ontology)
- pilot and Label Studio tooling

That matters because research pipelines tend to rot under iteration; tests keep the pipeline reproducible as it scales.

---

## 12) How to Run It (Minimal)

### Backend

```bash
pip install -r requirements.txt
uvicorn src.api.main:app --reload
```

Then open:

- API: `http://localhost:8000`
- Docs: `http://localhost:8000/docs`

### Frontend (optional)

```bash
cd qbm-frontend
npm install
npm run dev
```

Configure `qbm-frontend/.env.local` (see `qbm-frontend/.env.example`) to point to your backend and Thesys key.

---

## 13) What You Built (in one sentence)

You built an end-to-end scholarly research stack that turns the entire Quran into a **behaviorally-indexed dataset**, validates it like a research instrument, enriches it with **multi-tafsir evidence**, and makes it explorable through a **modern API + generative research UI + GPU-powered discovery** layer.
