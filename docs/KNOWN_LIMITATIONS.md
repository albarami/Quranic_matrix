# QBM Known Limitations

## Overview

This document explicitly lists cases where the QBM system refuses to answer, known classification challenges, and areas requiring human expertise.

## System Refuses to Answer

### 1. No Evidence Found

**Behavior**: Returns empty proof with `no_evidence: true`

**Triggers**:
- Query about topics not covered in Quran (e.g., modern technology specifics)
- Misspelled Arabic terms that don't match vocabulary
- Queries in unsupported languages

**Response**:
```json
{
  "answer": "لم يتم العثور على دليل قرآني لهذا السؤال",
  "proof": {
    "quran": [],
    "no_evidence": true
  }
}
```

### 2. Ambiguous Query

**Behavior**: Returns partial results with `ambiguous: true`

**Triggers**:
- Single-word queries with multiple meanings
- Queries mixing unrelated concepts

**Recommendation**: User should refine query with more context.

### 3. Controversial Topics

**Behavior**: Returns evidence without interpretation

**Triggers**:
- Sectarian-specific questions
- Disputed fiqh matters
- Political interpretations

**Response**: Raw evidence only, no synthesized answer. Defers to scholar review.

## Classification Challenges

### 1. Implicit Agents

**Challenge**: Many Quranic commands have implicit agents.

**Example**: "أَقِيمُوا الصَّلَاةَ" (Establish prayer)
- Explicit agent: None in text
- Implied agent: Believers

**Current Approach**: Default to `AGT_BELIEVER` for commands, flag for review.

### 2. Metaphorical Language

**Challenge**: Distinguishing literal vs. metaphorical behaviors.

**Example**: "قُلُوبُهُمْ كَالْحِجَارَةِ" (Hearts like stones)
- Literal: Physical heart
- Metaphorical: Spiritual hardness

**Current Approach**: Use tafsir context to determine. Flag ambiguous cases.

### 3. Multi-Behavior Spans

**Challenge**: Single verse may contain multiple behaviors.

**Example**: "وَأَقِيمُوا الصَّلَاةَ وَآتُوا الزَّكَاةَ" (Establish prayer and give zakat)
- Behavior 1: Prayer (worship)
- Behavior 2: Zakat (financial)

**Current Approach**: Create separate spans with shared verse reference.

### 4. Conditional Behaviors

**Challenge**: Behaviors with complex conditions.

**Example**: "إِن كُنتُمْ تُحِبُّونَ اللَّهَ فَاتَّبِعُونِي" (If you love Allah, follow me)
- Condition: Love of Allah
- Behavior: Following Prophet

**Current Approach**: Mark as `conditional: true`, extract both parts.

### 5. Negated Behaviors

**Challenge**: Prohibitions vs. descriptions of non-believers.

**Example**: "لَا تَقْرَبُوا الصَّلَاةَ وَأَنتُمْ سُكَارَىٰ" (Don't approach prayer while intoxicated)
- Prohibition: Prayer while intoxicated
- Not: General prohibition of prayer

**Current Approach**: Parse negation scope carefully, use tafsir for context.

## Tafsir Limitations

### 1. Missing Tafsir

**Issue**: Some verses lack tafsir in one or more sources.

**Affected**: ~5% of verses have incomplete tafsir coverage.

**Handling**: Return available sources, note missing ones.

### 2. Tafsir Disagreements

**Issue**: Classical scholars sometimes disagree.

**Example**: Interpretation of "الصِّرَاطَ الْمُسْتَقِيمَ" (Straight Path)
- Some: Islam generally
- Some: Specific practices
- Some: Following Sunnah

**Handling**: Present all views, don't synthesize single answer.

### 3. Language Variations

**Issue**: Classical Arabic vs. modern understanding.

**Example**: Words with shifted meanings over centuries.

**Handling**: Use classical tafsir definitions, not modern dictionaries.

## Graph Limitations

### 1. Incomplete Relationships

**Issue**: Not all valid relationships are captured.

**Current Coverage**: ~4460 edges covering major relationships.

**Missing**: Rare or subtle connections.

### 2. Confidence Calibration

**Issue**: Confidence scores are heuristic, not probabilistic.

**Interpretation**: 
- 0.9+ = Strong scholarly consensus
- 0.7-0.9 = Majority view
- 0.5-0.7 = Plausible but debated
- <0.5 = Weak or minority view

### 3. Causal vs. Correlation

**Issue**: Some edges represent correlation, not causation.

**Example**: Patience and reward often co-occur, but the causal mechanism is complex.

**Handling**: Edge types distinguish `CAUSES` from `ASSOCIATED_WITH`.

## Retrieval Limitations

### 1. Semantic Gap

**Issue**: User query may use different terms than source text.

**Example**: User asks about "stress" but Quran uses "ضيق" or "حرج".

**Mitigation**: Synonym expansion, but not comprehensive.

### 2. Cross-Lingual Queries

**Issue**: English queries may miss Arabic-specific nuances.

**Example**: "Patience" maps to صبر, but Arabic has richer vocabulary.

**Mitigation**: Bilingual vocabulary mapping, but imperfect.

### 3. Long Queries

**Issue**: Very long queries may dilute relevance.

**Recommendation**: Keep queries focused, <50 words.

## Performance Limitations

### 1. Cold Start

**Issue**: First query after startup is slower.

**Cause**: Loading data files, building indexes.

**Mitigation**: Warm-up endpoint, pre-loading.

### 2. Large Result Sets

**Issue**: SURAH_REF queries for long surahs return many results.

**Example**: Surah Al-Baqarah has 286 verses.

**Mitigation**: Summary mode, pagination, lazy loading.

### 3. Graph Traversal Depth

**Issue**: Deep traversals are expensive.

**Limit**: Max depth of 3 by default.

**Recommendation**: Use targeted queries instead of broad traversals.

## Data Quality Limitations

### 1. OCR Artifacts

**Issue**: Some tafsir texts have OCR errors from digitization.

**Affected**: ~1% of chunks may have minor errors.

**Handling**: Flag suspicious text, scholar review.

### 2. Diacritics Inconsistency

**Issue**: Arabic diacritics (tashkeel) vary across sources.

**Handling**: Normalize for search, preserve original for display.

### 3. Verse Numbering

**Issue**: Minor variations in verse numbering across traditions.

**Standard**: Using Uthmani numbering as canonical.

## Areas Requiring Human Expertise

### 1. Fatwa-Level Questions

**Not Suitable For**:
- "Is X halal or haram?"
- "What is the ruling on Y?"

**Reason**: Requires qualified scholar, not AI.

**Response**: Provide evidence, recommend consulting scholar.

### 2. Personal Spiritual Guidance

**Not Suitable For**:
- "What should I do about my faith crisis?"
- "How do I become a better Muslim?"

**Reason**: Requires pastoral care, not information retrieval.

**Response**: Provide relevant verses, recommend speaking with imam.

### 3. Interfaith Comparisons

**Not Suitable For**:
- "Is Islam better than Christianity?"
- "What does Quran say about other religions?"

**Reason**: Sensitive, requires nuanced scholarly treatment.

**Response**: Provide direct Quranic references only, no commentary.

### 4. Historical Disputes

**Not Suitable For**:
- Questions about specific historical events
- Sectarian historical narratives

**Reason**: Beyond scope of behavioral classification.

**Response**: Defer to historical scholarship.

## Reporting Issues

If you encounter:
- Incorrect classifications
- Missing relationships
- Tafsir errors
- System bugs

Please submit a scholar review through the `/reviews` interface or contact the development team.

## proof_only Mode Limitations (v2.0.1+)

When using `proof_only=true`:

1. **No LLM Answer**: The `answer` field contains a placeholder, not a generated response
2. **Structured Intents Only**: Works best with SURAH_REF and AYAH_REF queries
3. **No Semantic Search**: FREE_TEXT queries return empty results (use full mode)
4. **No Graph Traversal**: Graph evidence is not included
5. **Quran Text**: May show placeholder text if source file not available

This mode is designed for fast evidence retrieval, not complete proof generation.

## 7-Source Tafsir Guarantee

For structured intents (SURAH_REF, AYAH_REF), the system guarantees retrieval from all 7 tafsir sources:
- ibn_kathir, tabari, qurtubi, saadi, jalalayn, baghawi, muyassar

This guarantee does NOT apply to:
- FREE_TEXT queries (semantic search may not find all sources)
- CONCEPT_REF queries (depends on concept coverage)
- Verses with genuinely missing tafsir in source texts

## Version

This document reflects QBM v2.0.1 limitations. Future versions may address some of these issues.
