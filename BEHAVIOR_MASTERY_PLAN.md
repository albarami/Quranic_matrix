# QBM Behavior Mastery Implementation Plan

**Version:** 1.0.0  
**Date:** 2026-01-01  
**Goal:** Transform QBM from audit-grade retrieval to **behavioral mastery brain**

---

## Executive Summary

Passing 200/200 benchmark proves internal consistency, NOT scholar-level intelligence. True mastery requires:

1. **Complete dossiers** for all 87 behaviors (vocab, evidence, relationships, Bouzidani contexts)
2. **Unified graph schema** (no type drift: "behavior" vs "BEHAVIOR")
3. **Discovery engine** (multi-hop paths, bridges, motifs, communities)
4. **Mastery-driven query answering** (dossier-first, not ad-hoc retrieval)
5. **Expanded evaluation** (1000+ mastery questions, not just 200 smoke tests)

---

## Non-Negotiables

| Rule | Description |
|------|-------------|
| **87 behaviors** | Canonical truth stays at 87. No relaxing to 73 anywhere. |
| **No hallucinations** | Every claim must be (A) directly evidenced, or (B) deterministic derivation with rule ID |
| **Single source of truth** | One registry for entity IDs and vocab mapping. No duplicates. |
| **Test-gated changes** | Add tests before claiming completion. CI must pass. |

---

## Phase 1: Graph Normalization Layer

**Goal:** Eliminate schema drift. One contract, one loader, one normalized output.

### Deliverables

| File | Description |
|------|-------------|
| `src/graph/__init__.py` | Package init |
| `src/graph/normalize.py` | `normalize_graph()` function + `GraphV3` schema |
| `src/graph/contract.py` | Graph contract validation |
| `tests/test_graph_normalization.py` | Normalization tests |

### Schema Contract (GraphV3)

```python
@dataclass
class GraphV3:
    nodes: List[NodeV3]  # 87 behaviors minimum
    edges: List[EdgeV3]  # typed edges with evidence
    metadata: GraphMetadata

@dataclass
class NodeV3:
    id: str              # BEH_*, AGT_*, ORG_*, STA_*
    node_type: str       # BEHAVIOR, AGENT, ORGAN, STATE (uppercase)
    label_ar: str
    label_en: str
    attributes: Dict[str, Any]

@dataclass
class EdgeV3:
    source: str
    target: str
    edge_type: str       # CAUSES, LEADS_TO, PREVENTS, STRENGTHENS, OPPOSITE
    evidence_count: int
    evidence_keys: List[str]  # verse keys
    confidence: float
```

### Tests Required

```python
# tests/test_graph_normalization.py
def test_normalize_graph_has_87_behaviors():
    """Normalized graph must have exactly 87 behavior nodes."""

def test_normalize_graph_node_types_uppercase():
    """All node_type values must be uppercase."""

def test_normalize_graph_edge_types_valid():
    """All edge_type values must be in allowed set."""

def test_normalize_graph_edges_have_evidence():
    """All edges must have evidence_count >= 0 and evidence_keys list."""

def test_normalize_graph_deterministic():
    """Same input produces identical output (hash check)."""

def test_all_graph_files_normalize_successfully():
    """Load every graph file and confirm normalization succeeds."""
```

### Acceptance Criteria

- [ ] `normalize_graph()` handles both old and new schema formats
- [ ] All planners/engines use normalized graph (single entry point)
- [ ] 87 behavior nodes present after normalization
- [ ] All tests pass
- [ ] Git commit + push

---

## Phase 2: Behavior Mastery Builder

**Goal:** Generate complete dossier for each of 87 behaviors.

### Deliverables

| File | Description |
|------|-------------|
| `scripts/build_behavior_mastery.py` | Main builder script |
| `src/mastery/__init__.py` | Package init |
| `src/mastery/dossier.py` | `BehaviorDossier` dataclass + builder |
| `src/mastery/assembler.py` | Merges all sources into dossiers |
| `artifacts/mastery/behaviors/*.json` | 87 dossier files |
| `artifacts/mastery/mastery_summary.json` | Coverage stats |
| `artifacts/mastery/mastery_manifest.json` | Hashes + schema version |
| `tests/test_mastery_dossiers.py` | Dossier validation tests |

### Dossier Schema

```python
@dataclass
class BehaviorDossier:
    # Identity
    behavior_id: str           # BEH_EMO_PATIENCE
    label_ar: str              # الصبر
    label_en: str              # Patience
    roots: List[str]           # صبر
    stems: List[str]           # صابر، صبور، اصطبر
    
    # Vocabulary Mastery
    synonyms_ar: List[str]
    synonyms_en: List[str]
    antonyms: List[str]
    morphological_variants: List[str]
    quranic_phrases: List[str]
    
    # Evidence Mastery
    quran_evidence: List[VerseEvidence]  # verse_key, text, relevance (direct/indirect)
    tafsir_evidence: Dict[str, List[TafsirChunk]]  # per source
    
    # Relationship Mastery
    outgoing_edges: List[RelationshipEdge]  # this behavior -> other
    incoming_edges: List[RelationshipEdge]  # other -> this behavior
    causal_paths: List[CausalPath]          # multi-hop paths with evidence
    
    # Bouzidani Context Matrix
    bouzidani_contexts: BouzidaniContexts
    
    # Metadata
    evidence_stats: EvidenceStats
    completeness_score: float
    missing_fields: List[str]
    generated_at: str
    schema_version: str

@dataclass
class BouzidaniContexts:
    # Axis 1: Organic/Biological
    organ_links: List[str]           # ORG_HEART, ORG_TONGUE, etc.
    internal_external: str           # باطن, ظاهر, both, unknown
    
    # Axis 2: Situational
    situational_context: str         # النفس, الآفاق, الخالق, الكون, الحياة, unknown
    
    # Axis 3: Systemic
    systemic_context: str            # البيت, العمل, مكان عام, unknown
    
    # Axis 4: Spatial
    spatial_context: str             # location-based or unknown
    
    # Axis 5: Temporal
    temporal_context: str            # صباحاً, ظهراً, عصراً, ليلاً, unknown
    
    # Additional
    intention_niyyah: str            # with_intention, without_intention, instinctive, unknown
    recurrence_dawrah: str           # one_time, periodic, continuous, unknown
    
    # Reason codes for unknowns
    unknown_reasons: Dict[str, str]
```

### Tests Required

```python
# tests/test_mastery_dossiers.py
def test_mastery_dossier_count_is_87():
    """Exactly 87 dossier files must exist."""

def test_each_dossier_has_required_fields():
    """Every dossier must have all required schema fields."""

def test_each_relationship_has_evidence():
    """Every relationship edge must have evidence_count > 0 or explicit reason."""

def test_bouzidani_context_fields_present_or_explicit_unknown():
    """All Bouzidani context fields must be populated or explicitly 'unknown' with reason."""

def test_mastery_manifest_hashes_reproducible():
    """Running builder twice produces identical hashes."""

def test_dossier_schema_validation():
    """Each dossier validates against JSON schema."""

def test_quran_evidence_has_verse_text():
    """Each Quran evidence entry must include actual verse text."""

def test_tafsir_evidence_has_provenance():
    """Each tafsir chunk must have source, chunk_id, and offsets."""
```

### Acceptance Criteria

- [ ] 87/87 dossiers generated
- [ ] Each dossier validates schema
- [ ] Missing fields explicitly marked as `unknown` with reason codes
- [ ] `mastery_summary.json` shows coverage stats
- [ ] `mastery_manifest.json` has reproducible hashes
- [ ] All tests pass
- [ ] Git commit + push

---

## Phase 3: Discovery Engine

**Goal:** Deterministic algorithms for multi-hop insights beyond human capacity.

### Deliverables

| File | Description |
|------|-------------|
| `src/graph/discovery.py` | Discovery algorithms |
| `src/graph/hypothesis.py` | Hypothesis dataclass + validation |
| `scripts/build_discovery_report.py` | Generate discovery artifacts |
| `artifacts/discovery/discovery_report.json` | Ranked hypotheses |
| `artifacts/discovery/bridges.json` | Bridge behaviors |
| `artifacts/discovery/communities.json` | Behavior clusters |
| `artifacts/discovery/motifs.json` | Repeated subgraph patterns |
| `tests/test_discovery_engine.py` | Discovery tests |

### Discovery Algorithms

```python
class DiscoveryEngine:
    def find_multihop_paths(self, min_hops=2, max_hops=6, top_k=100) -> List[CausalPath]:
        """Enumerate top-k paths between behaviors with evidence."""
    
    def find_bridge_behaviors(self, top_k=20) -> List[BridgeBehavior]:
        """Compute betweenness centrality to identify connectors."""
    
    def detect_communities(self) -> List[BehaviorCommunity]:
        """Cluster behaviors and label clusters using evidence terms."""
    
    def find_motifs(self, min_support=3) -> List[GraphMotif]:
        """Find repeated subgraph patterns (triads, etc.)."""
    
    def predict_links(self, min_confidence=0.7) -> List[LinkHypothesis]:
        """Propose candidate edges (labeled as hypotheses, not facts)."""
```

### Hypothesis Schema

```python
@dataclass
class Hypothesis:
    hypothesis_id: str
    hypothesis_type: str           # path, bridge, motif, link_prediction
    involved_behaviors: List[str]  # canonical IDs
    evidence_bundle: EvidenceBundle
    confidence_score: float        # deterministic rubric
    confidence_factors: Dict[str, float]  # evidence_count, source_diversity, etc.
    falsification_check: FalsificationResult
    generated_at: str

@dataclass
class FalsificationResult:
    counter_evidence_found: bool
    counter_evidence: List[str]    # verse keys if found
    search_scope: str              # "full_corpus" or limited
```

### Tests Required

```python
# tests/test_discovery_engine.py
def test_discovery_report_reproducible():
    """Same input produces identical discovery report."""

def test_multihop_paths_have_evidence():
    """Every path edge must have evidence."""

def test_bridge_behaviors_are_valid():
    """All bridge behaviors must be in canonical 87."""

def test_communities_cover_all_behaviors():
    """Every behavior must belong to exactly one community."""

def test_motifs_have_minimum_support():
    """Each motif must appear at least min_support times."""

def test_hypotheses_have_falsification_check():
    """Every hypothesis must include falsification result."""

def test_link_predictions_labeled_as_hypotheses():
    """Link predictions must not be presented as facts."""
```

### Acceptance Criteria

- [ ] Discovery report generated deterministically
- [ ] All hypotheses have evidence bundles
- [ ] All hypotheses have falsification checks
- [ ] Bridge behaviors identified with centrality scores
- [ ] Communities detected and labeled
- [ ] Motifs found with support counts
- [ ] All tests pass
- [ ] Git commit + push

---

## Phase 4: Mastery API Endpoints + Runtime Integration

**Goal:** Serve mastery data via API. Query answering uses dossiers first.

### Deliverables

| File | Description |
|------|-------------|
| `src/mastery/engine.py` | `BehaviorMasteryEngine` class |
| `src/api/routers/mastery.py` | Mastery API endpoints |
| `tests/test_mastery_api.py` | API tests |

### API Endpoints

```
GET  /api/behaviors                      # List all 87 behaviors
GET  /api/behaviors/{id}                 # Get behavior summary
GET  /api/behaviors/{id}/mastery         # Get full dossier
GET  /api/behaviors/{id}/relationships   # Get relationship graph
GET  /api/behaviors/{id}/evidence        # Get evidence bundle
GET  /api/discovery/bridges              # Get bridge behaviors
GET  /api/discovery/communities          # Get behavior clusters
GET  /api/discovery/paths?from=X&to=Y    # Get paths between behaviors
POST /api/query                          # Mastery-aware query (routes to engine)
```

### Response Schema (with provenance)

```json
{
  "behavior_id": "BEH_EMO_PATIENCE",
  "answer": { ... },
  "provenance": {
    "dossier_version": "1.0.0",
    "dossier_hash": "abc123...",
    "evidence_chunks_used": ["2:153", "3:200", ...],
    "derivation_rules_triggered": ["RULE_CAUSAL_001", "RULE_CONTEXT_003"],
    "confidence_score": 0.92
  },
  "limitations": ["temporal_context unknown", "no spatial evidence found"]
}
```

### BehaviorMasteryEngine

```python
class BehaviorMasteryEngine:
    def __init__(self, mastery_dir: Path):
        self.dossiers = self._load_dossiers(mastery_dir)
        self.discovery = self._load_discovery(mastery_dir)
    
    def get_behavior_mastery(self, behavior_id: str) -> BehaviorDossier:
        """Get complete dossier for a behavior."""
    
    def answer(self, query: str) -> MasteryAnswer:
        """Answer query using dossier + allowed derivation rules."""
    
    def get_causal_neighborhood(self, behavior_id: str, depth: int = 2) -> SubGraph:
        """Get local causal graph around a behavior."""
    
    def explain_relationship(self, from_id: str, to_id: str) -> RelationshipExplanation:
        """Explain relationship with evidence and paths."""
```

### Tests Required

```python
# tests/test_mastery_api.py
def test_get_all_behaviors_returns_87():
    """GET /api/behaviors returns exactly 87 behaviors."""

def test_get_behavior_mastery_has_provenance():
    """Mastery response includes dossier_version and hash."""

def test_query_routes_to_mastery_engine():
    """Behavior queries use mastery engine, not generic fallback."""

def test_response_has_limitations_when_incomplete():
    """Response includes limitations for unknown fields."""

def test_derivation_rules_are_recorded():
    """Any derived claims include rule IDs in provenance."""
```

### Acceptance Criteria

- [ ] All mastery endpoints functional
- [ ] Responses include full provenance
- [ ] Query routing uses mastery engine for behavior queries
- [ ] Limitations explicitly stated
- [ ] All tests pass
- [ ] Git commit + push

---

## Phase 5: Expanded Evaluation (Mastery Benchmark)

**Goal:** 1000+ questions testing true mastery, not just retrieval.

### Deliverables

| File | Description |
|------|-------------|
| `data/benchmarks/qbm_mastery_1000.v1.jsonl` | Mastery benchmark |
| `src/benchmarks/mastery_scoring.py` | Mastery-specific scoring |
| `scripts/run_mastery_benchmark.py` | Benchmark runner |
| `tests/test_mastery_benchmark.py` | Benchmark tests |

### Question Categories (1000+ questions)

| Category | Count | Description |
|----------|-------|-------------|
| Behavior Resolution | 200 | Same behavior asked in multiple forms (Arabic variants, synonyms) |
| Context Queries | 150 | Organ/internal-external/social/spatial/temporal contexts |
| Causal Paths | 150 | Multi-hop causal chains with evidence |
| Prevention/Strengthening | 100 | What prevents X? What strengthens Y? |
| Comparative | 100 | Compare behavior A vs B |
| Discovery | 100 | Bridge behaviors, communities, motifs |
| Falsification | 100 | Queries designed to test fail-closed behavior |
| Edge Cases | 100 | Unknown contexts, missing evidence, ambiguous queries |

### Scoring Criteria

```python
@dataclass
class MasteryScore:
    behavior_resolution_correct: bool
    required_sections_present: bool
    provenance_complete: bool
    no_hallucinated_claims: bool
    limitations_stated: bool
    derivation_rules_valid: bool
    
    def verdict(self) -> str:
        if self.no_hallucinated_claims is False:
            return "FAIL_HALLUCINATION"
        if not all([self.behavior_resolution_correct, 
                    self.required_sections_present,
                    self.provenance_complete]):
            return "PARTIAL"
        return "PASS"
```

### Tests Required

```python
# tests/test_mastery_benchmark.py
def test_benchmark_has_1000_plus_questions():
    """Mastery benchmark must have >= 1000 questions."""

def test_all_categories_represented():
    """All question categories must have minimum count."""

def test_scoring_detects_hallucinations():
    """Scoring must flag fabricated claims as FAIL_HALLUCINATION."""

def test_benchmark_reproducible():
    """Same questions produce same scores."""
```

### Acceptance Criteria

- [ ] 1000+ questions across all categories
- [ ] Scoring detects hallucinations (0 tolerance)
- [ ] ≥95% PASS rate required for release
- [ ] 0 FAIL_HALLUCINATION required for release
- [ ] All tests pass
- [ ] Git commit + push

---

## Phase 6: Go-Live Validation + Audit Pack

**Goal:** Complete operational readiness with full audit trail.

### Deliverables

| File | Description |
|------|-------------|
| `scripts/preflight_mastery_release.py` | Pre-flight checks for mastery |
| `scripts/generate_mastery_audit_pack.py` | Generate audit artifacts |
| `artifacts/audit/mastery_audit_pack.zip` | Complete audit package |

### Go-Live Command Sequence

```powershell
# 1. Set environment
$env:QBM_DATASET_MODE = "full"
$env:QBM_USE_FIXTURE = "0"
$env:PYTHONUTF8 = "1"

# 2. Run preflight checks
python scripts/preflight_full_release.py

# 3. Build mastery artifacts
python scripts/build_behavior_mastery.py
python scripts/build_discovery_report.py

# 4. Run original benchmark (must be 200/200 PASS)
python scripts/run_qbm_benchmark.py --dataset data/benchmarks/qbm_legendary_200.v1.jsonl --ci

# 5. Run mastery benchmark (must be ≥95% PASS, 0 hallucinations)
python scripts/run_mastery_benchmark.py --dataset data/benchmarks/qbm_mastery_1000.v1.jsonl --ci

# 6. Generate audit pack
python scripts/generate_mastery_audit_pack.py --strict

# 7. Start production server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Mastery Gates Summary

| Gate | Requirement | Blocks Release |
|------|-------------|----------------|
| Graph Normalization | 87 behaviors, valid types | Yes |
| Behavior Dossiers | 87/87 complete | Yes |
| Discovery Report | Reproducible, evidence-backed | Yes |
| Original Benchmark | 200/200 PASS | Yes |
| Mastery Benchmark | ≥95% PASS | Yes |
| Hallucination Check | 0 fabricated claims | Yes |
| Audit Pack | All artifacts present | Yes |

### Acceptance Criteria

- [ ] All preflight checks pass
- [ ] Mastery artifacts built and validated
- [ ] Original benchmark: 200/200 PASS
- [ ] Mastery benchmark: ≥95% PASS, 0 hallucinations
- [ ] Audit pack generated with all artifacts
- [ ] API serves mastery endpoints
- [ ] All tests pass
- [ ] Git commit + push

---

## Implementation Timeline

| Phase | Description | Estimated Effort |
|-------|-------------|------------------|
| Phase 1 | Graph Normalization | 1 session |
| Phase 2 | Behavior Mastery Builder | 2 sessions |
| Phase 3 | Discovery Engine | 2 sessions |
| Phase 4 | Mastery API + Runtime | 1 session |
| Phase 5 | Expanded Evaluation | 2 sessions |
| Phase 6 | Go-Live Validation | 1 session |

---

## Success Definition

**"Super Smart" = All of the following:**

1. ✅ 87/87 behavior dossiers complete with Bouzidani contexts
2. ✅ Unified graph schema (no drift)
3. ✅ Discovery engine produces reproducible insights
4. ✅ Mastery API serves dossiers with full provenance
5. ✅ 200/200 original benchmark PASS
6. ✅ ≥95% mastery benchmark PASS
7. ✅ 0 hallucinated claims
8. ✅ Every claim is evidence-backed or derivation-ruled
9. ✅ Audit pack validates all artifacts

---

## Contact

**System:** Quranic Behavioral Matrix (QBM)  
**Framework:** Bouzidani Five-Context Framework (مصفوفة بوزيداني للسياقات الخمسة)
