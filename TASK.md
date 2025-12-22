# TASKS

## Active

- (none)

## QBM AI System Implementation (Phases 1-4 Complete)

| Phase | Status | Tests | Git Tag |
|-------|--------|-------|---------|
| 1. Knowledge Graph | ✅ Complete | 29 | v0.1.0-graph |
| 2. Vector Store + RAG | ✅ Complete | 29 | v0.2.0-rag |
| 3. Ontology (RDFLib) | ✅ Complete | 18 | v0.3.0-ontology |
| 4. Multi-Tafsir | ✅ Complete | 16 | v0.4.0-tafsir |
| **Total** | **4/8 Phases** | **92 tests** | |

### Implemented Modules (`src/ai/`)
- `graph/qbm_graph.py` - NetworkX + SQLite knowledge graph (73 behaviors, 111 edges)
- `vectors/qbm_vectors.py` - ChromaDB vector store with AraBERT embeddings
- `rag/qbm_rag.py` - RAG pipeline with Azure OpenAI (GPT-5)
- `ontology/qbm_ontology.py` - RDFLib OWL ontology with SPARQL
- `tafsir/cross_tafsir.py` - Multi-tafsir analysis (7 sources)

### Remaining Phases
- Phase 5: Complete Annotation Pipeline
- Phase 6: Model Fine-Tuning
- Phase 7: Discovery System
- Phase 8: Integration & Deployment

## Completed

- **2025-12-20**: Align Label Studio task JSON format to `{"id":..., "data":{...}}` (pilot + sample tasks); add tests; push to GitHub.
- **2025-12-20**: Add pilot_50 tokenized selection dataset + validation/build tooling; push to GitHub.
- **2025-12-20**: Create Qur’anic Human‑Behavior Classification Matrix documentation files (`.md` + `.doc`/RTF) and place under `docs/`.
- **2025-12-20**: Bootstrap startup requirements (docs + vocabularies + data/schema stubs) and push to GitHub.
- **2025-12-20**: Add Label Studio configuration + export converter + tests, and push to GitHub.
- **2025-12-20**: Add Coding Manual v1.0 (annotator training) + gold standard examples placeholder; push to GitHub.
- **2025-12-20**: Add Gold Standard Examples v1.0 (20 calibration examples) and push to GitHub.
- **2025-12-20**: Place full Qur'an tokenization artifact into `data/quran/` in contract format; add converter + tests.
- **2025-12-20**: Set up Label Studio with pilot_50 tasks; create API annotation script; annotate all 50 pilot ayat programmatically.
- **2025-12-20**: Achieve 100% accuracy against gold standards (10 overlapping examples); fix deontic_signal, behavior_form, agent_type errors.
- **2025-12-20**: Add validation scripts to `src/validation/` (validate_schema.py, validate_vocabularies.py, calculate_iaa.py).
- **2025-12-20**: Phase 0 and Phase 1 COMPLETE. Ready for Phase 2 Micro-Pilot.

## Discovered During Work

- Label Studio SDK requires enabling legacy API tokens in database (`jwt_auth_jwtsettings.legacy_api_tokens_enabled = 1`)
- JSONL format not supported for Label Studio import; must convert to JSON array


