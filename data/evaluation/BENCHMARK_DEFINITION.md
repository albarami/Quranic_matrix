# QBM Embedding Benchmark Definition

## Decision: Relatedness Benchmark (not Equivalence)

### Rationale
The QBM system's primary use case is **retrieval**: given a query about a behavior (e.g., "الصبر"), 
find relevant verses, tafsir passages, and related concepts. This is a **relatedness** task, not 
a strict paraphrase/equivalence task.

For retrieval:
- Antonyms ARE related (same semantic field, opposite meaning)
- Complementary concepts ARE related (e.g., hope/fear in Islamic spirituality)
- Only truly unrelated concepts (different semantic field entirely) should score low

### Benchmark Structure

#### gold_relatedness.jsonl (PRIMARY - for retrieval evaluation)
Labels:
- `1.0` = equivalent (same meaning: term ↔ definition, synonyms)
- `0.5` = related (same semantic field: antonyms, complementary, co-occurring)
- `0.0` = unrelated (different semantic field entirely)

Categories:
- `same_concept`: term and its definition → 1.0
- `synonym`: synonymous terms → 1.0
- `opposite`: antonyms (same field, opposite meaning) → 0.5
- `complementary`: theologically paired concepts → 0.5
- `unrelated`: different semantic field → 0.0

#### gold_equivalence_v2.jsonl (SECONDARY - for paraphrase detection)
Labels:
- `1.0` = equivalent (same meaning)
- `0.0` = not equivalent (everything else)

Categories:
- `same_concept`: term and definition → 1.0
- `synonym` / `near_synonym`: synonymous terms → 1.0
- `opposite`: antonyms → 0.0 (not same meaning)
- `not_equivalent`: related but distinct → 0.0
- `hard_not_equivalent`: confusable neighbors within Islamic vocabulary → 0.0
- `unrelated`: different domains → 0.0

This is stricter: antonyms and related concepts score 0.0 because they don't mean the same thing.
Includes hard non-equivalents (e.g., صبر↔توكل, حسد↔غيرة) to prevent trivial wins.

### Evaluation Metrics

For **relatedness** (primary):
- Spearman correlation (graded ranking)
- Separation: avg(equivalent) > avg(related) > avg(unrelated)
- Retrieval: Recall@k, nDCG@k, MRR

For **equivalence** (secondary):
- AUC (binary classification)
- Accuracy on high/low pairs

### Canonical Retrieval Substrate

**Decision: Full Power System (TorchGPUVectorSearch) is canonical**

The Full Power system uses:
- `TorchGPUVectorSearch` for vector similarity
- `GPUEmbeddingPipeline` for embeddings
- `CrossEncoderReranker` for reranking

ChromaDB (`QBMVectorStore`) exists but is NOT in the critical path for the proof system.
RAG should use the Full Power index, not ChromaDB.

### Training Data Construction

Valid positive pairs:
1. **Definition-anchored**: (term, definition) - e.g., ("الصبر", "الثبات عند البلاء")
2. **Synonym-anchored**: (term, synonym) - e.g., ("الصبر", "التحمل")
3. **Verse-tafsir aligned**: (verse_text, tafsir_explaining_verse)

Valid negative pairs:
1. **Unrelated**: (term, unrelated_concept) - different semantic field
2. **Hard negatives**: (term, same_topic_different_meaning) - for fine-grained discrimination

### Benchmark Size

**Smoke Benchmark (current)**: ~58 pairs in `gold_relatedness_v2.jsonl`
- Fixed domain semantics (split compound pairs like الكبر↔التواضع vs الخشوع)
- Hard unrelateds within Islamic vocabulary (not trivial like صبر↔أكل)
- Confusable neighbors (حسد↔غيرة, صبر↔توكل)
- Sufficient for rapid iteration and gating
- Not sufficient for final model selection

**V1 Benchmark (planned)**: 200-500 pairs
- Required before production deployment
- **Must include verse↔tafsir aligned pairs** (user to provide examples)
- Should cover all 87 behaviors in taxonomy
- Should include more confusable neighbors for fine-grained discrimination

---

## Acceptance Criteria (Phase 5)

### 1. Category-Order Invariant (HARD GATE)
```
mean(sim(equivalent)) > mean(sim(related)) > mean(sim(unrelated))
```
Each gap must be ≥ 0.15

### 2. Pairwise Win-Rate ≥ 70%
- % of times equivalent pair scores higher than related pair
- % of times related pair scores higher than unrelated pair

### 3. Spearman ≥ 0.35 on smoke benchmark
(Increase to ≥ 0.50 when V1 benchmark is ready)

### 4. Retrieval on Full Power Pipeline
Using `TorchGPUVectorSearch` with production index and preprocessing:
- Recall@10 ≥ 0.70 on verse↔tafsir pairs
- nDCG@10 ≥ 0.60
- Report metrics both with and without `CrossEncoderReranker`

### 5. No Synthetic Evidence (HARD FAIL)
- Proof output must carry: `synthetic_evidence_used: false`
- Tests fail if any synthetic edges/weights are returned
- "No evidence found" is acceptable; fabricated evidence is not

### 6. Reproducibility
Same model + same index + same benchmark → same metrics (within ±0.01 tolerance)
