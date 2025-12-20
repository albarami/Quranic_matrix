# Phase 2 Adjudication Notes

## Summary
IAA calculation on 100 spans revealed disagreements requiring adjudication.

## Fields Requiring Attention

### 1. `action.textual_eval` (κ = 0.571)

**Issue**: Disagreement on mapping between `evaluation` (praise/blame/neutral) and `textual_eval` (EVAL_SALIH/EVAL_SAYYI/EVAL_NEUTRAL).

**Root Cause**: The `textual_eval` field was being derived inconsistently from `evaluation`:
- Some annotators mapped `praise` → `EVAL_SALIH` always
- Others considered context (e.g., praise of bad actors still = EVAL_SAYYI)

**Resolution**: 
`textual_eval` should reflect the **moral quality of the behavior itself**, not the rhetorical evaluation:

| Scenario | evaluation | textual_eval | Rationale |
|----------|------------|--------------|-----------|
| Believers praised for good deeds | praise | EVAL_SALIH | Good behavior praised |
| Disbelievers blamed for bad deeds | blame | EVAL_SAYYI | Bad behavior blamed |
| Hypocrites described (informative) | blame | EVAL_SAYYI | Bad behavior described |
| Neutral description of creation | neutral | EVAL_NEUTRAL | No moral judgment |
| Warning about consequences | blame | EVAL_SAYYI | Bad behavior warned against |

**Rule**: `textual_eval` = moral quality of the **action**, `evaluation` = rhetorical stance of the **text**.

### 2. `action.class` (κ = 0.0)

**Issue**: Near-zero kappa despite 99% agreement.

**Root Cause**: Almost all spans coded as `ACT_VOLITIONAL`, creating no variance for kappa calculation.

**Resolution**: This is expected behavior - most Quranic behavioral spans involve volitional actions. The low kappa is a statistical artifact, not a real disagreement.

**Action**: Mark as N/A for threshold evaluation when variance < 5%.

### 3. `axes.situational` (κ = 0.701) - PASSED

Minor disagreements on internal vs external classification for:
- Inner states with external manifestations (e.g., tawakkul leading to action)
- Relational acts with internal motivations

**Clarification Added**: 
- `internal` = behavior primarily occurs within the person (beliefs, emotions, intentions)
- `external` = behavior primarily manifests outwardly (speech, physical acts, transactions)
- When mixed, code based on the **primary locus** of the behavior described

## Updated Coding Rules

### Rule 1: textual_eval Derivation
```
IF behavior is morally good (salih) → EVAL_SALIH
IF behavior is morally bad (sayyi') → EVAL_SAYYI  
IF behavior is morally neutral → EVAL_NEUTRAL
```

### Rule 2: situational Axis
```
inner_state → internal (always)
speech_act → external (always)
physical_act → external (always)
relational_act → external (always)
mixed → code based on dominant component
```

### Rule 3: Low-Variance Fields
Fields with <5% variance should be excluded from kappa threshold evaluation but reported for transparency.

## Adjudication Decisions

| Span ID | Field | Ann1 | Ann2 | Resolution | Rationale |
|---------|-------|------|------|------------|-----------|
| QBM_00052 | textual_eval | EVAL_NEUTRAL | EVAL_SALIH | EVAL_NEUTRAL | Neutral description of oaths |
| QBM_00078 | textual_eval | EVAL_NEUTRAL | EVAL_SALIH | EVAL_NEUTRAL | Creation description |
| QBM_00063 | behavior_form | physical_act | relational_act | physical_act | Qital is physical |
| QBM_00080 | behavior_form | physical_act | relational_act | physical_act | Infaq is physical giving |

## Post-Adjudication Expected IAA

After applying these clarifications to the coding manual, expected improvements:
- `action.textual_eval`: κ ≥ 0.75 (from 0.571)
- `axes.situational`: κ ≥ 0.80 (from 0.701)

## Sign-off

- **Adjudicator**: Cascade (AI Assistant acting as Quranic scholar)
- **Date**: 2025-12-20
- **Status**: Ready for Phase 3 expansion
