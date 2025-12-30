# Reverted/Documented Patches

> **Date**: 2024-12-30
> **Phase**: 0 (Freeze + Baseline)
> **Context**: Behavior Evidence Engine Rebuild

## Background

During initial debugging of behavior analysis queries (e.g., "حلل سلوك الصبر"), routing patches were added to `src/ml/mandatory_proof_system.py` to direct analytical queries to the concept index. However, these patches were **masking data corruption**, not fixing it.

### The Problem

The concept_index_v2.jsonl contains **79% invalid data** for behaviors like الصبر (patience):
- Total verses indexed: 291
- Actually containing صبر root: 60 (21%)
- Invalid verses: 231 (79%)

Routing queries to this corrupt index produces scholar-embarrassing outputs.

## Patches Identified

### Patch 1: Analytical Question Class Check (Lines 962-970)

**Location**: `src/ml/mandatory_proof_system.py:962-970`

```python
# Check if concept was extracted for analytical question_class (even with FREE_TEXT intent)
# This handles cases like "حلل سلوك الصبر" where intent=FREE_TEXT but question_class=behavior_profile_11axis
question_class = route_result.get("question_class", "free_text")
concept_term = route_result.get("concept")

if not quran_results and concept_term and question_class != "free_text":
    concept_verses = get_verses_from_concept(concept_term)
    quran_results.extend(concept_verses)
    logging.info(f"[PROOF] Analytical question_class={question_class} with concept='{concept_term}': {len(quran_results)} verses from concept index")
```

**Purpose**: Route analytical queries with concepts to concept_index
**Problem**: Routes to CORRUPT concept_index_v2

### Patch 2: Analytical Question Classes Set (Lines 1006-1024)

**Location**: `src/ml/mandatory_proof_system.py:1006-1024`

```python
# Analytical question classes that should also trigger LegendaryPlanner
# (even when intent is FREE_TEXT)
analytical_question_classes = {
    "behavior_profile_11axis", "causal_chain", "cross_tafsir",
    "graph_metrics", "heart_state_analysis", "agent_behavior",
    "temporal_spatial", "consequence_analysis",
}

question_class = route_result.get("question_class", "free_text")

planner_results = None
planner_debug = None
if intent in benchmark_intents or question_class in analytical_question_classes:
    from src.ml.legendary_planner import get_legendary_planner
    planner = get_legendary_planner()
    planner_results, planner_debug = planner.query(question)
```

**Purpose**: Trigger LegendaryPlanner for analytical question classes
**Problem**: LegendaryPlanner then retrieves from CORRUPT concept_index

## Decision: Keep Patches, Fix Data

The routing logic itself is **correct**. The problem is the data it routes to.

**We do NOT revert these patches.** Instead:

1. **Phase 1-4**: Rebuild data foundation with validated concept_index_v3
2. **Phase 9**: Update system to use concept_index_v3 instead of v2

After Phase 9, these routing patches will direct queries to **validated** data.

## Verification After Rebuild

After concept_index_v3 is deployed, verify:

```bash
# Query should return ~100 validated patience verses
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "حلل سلوك الصبر في القرآن"}'
```

Expected:
- ~100 verses (not 291 with 79% invalid)
- Every verse contains ص-ب-ر root
- Evidence provenance included

## Commit History

| Date | Action | Commit |
|------|--------|--------|
| 2024-12-30 | Documented patches (kept, not reverted) | Phase 0 |
| TBD | Updated to use concept_index_v3 | Phase 9 |
