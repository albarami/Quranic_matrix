# Tafsir Consultation Protocol for QBM Annotation

## Overview

This protocol guides annotators on when and how to consult tafsir (Quranic exegesis) during the annotation process. Proper tafsir consultation ensures accurate behavioral classification while maintaining annotation consistency.

---

## 1. When to Consult Tafsir

### 1.1 Required Consultation (Must Check)

| Situation | Example | Why |
|-----------|---------|-----|
| **Agent identification unclear** | "الذين" without clear antecedent | Tafsir clarifies who is addressed |
| **Behavior type ambiguous** | Action could be inner state or physical | Tafsir provides context |
| **Sabab al-Nuzul needed** | Verse references specific event | Historical context affects classification |
| **Multiple valid readings** | Verse has different qira'at affecting meaning | Tafsir documents variants |

### 1.2 Recommended Consultation

- Complex legal (fiqh) verses
- Verses with disputed interpretations
- Metaphorical language (majaz)
- Verses with abrogation (naskh) considerations

### 1.3 Not Required

- Clear, unambiguous behavioral descriptions
- Verses with obvious agent and action
- When previous similar verses have established pattern

---

## 2. Tafsir Source Hierarchy

Consult sources in this order of priority:

### Primary Sources (Arabic)

| Priority | Source | Strength | Best For |
|----------|--------|----------|----------|
| 1 | **Ibn Kathir** (ابن كثير) | Comprehensive, hadith-based | General reference, agent identification |
| 2 | **Al-Tabari** (الطبري) | Linguistic depth, multiple opinions | Linguistic analysis, variant readings |
| 3 | **Al-Qurtubi** (القرطبي) | Fiqh implications | Legal/behavioral rulings |
| 4 | **Al-Sa'di** (السعدي) | Modern clarity | Quick understanding |

### Using the Lookup Tool

```bash
# Single ayah lookup
python tools/tafsir/tafsir_lookup.py --surah 2 --ayah 255

# Compare all sources
python tools/tafsir/tafsir_lookup.py --surah 2 --ayah 255 --compare

# Search for term
python tools/tafsir/tafsir_lookup.py --search "المنافقين"
```

---

## 3. Consultation Workflow

### Step 1: Initial Assessment
1. Read the verse in Arabic
2. Identify potential behavioral content
3. Note any ambiguities

### Step 2: Tafsir Lookup (if needed)
1. Use lookup tool for primary source (Ibn Kathir)
2. Focus on:
   - Who is being addressed (agent)
   - What action/state is described
   - Evaluation (praise/blame/neutral)
   - Any conditions or context

### Step 3: Document Findings
Record in annotation:
- Which tafsir was consulted
- Key insight that influenced decision
- Any disagreement between sources

---

## 4. Documentation Requirements

### 4.1 When Tafsir Influenced Decision

Add to annotation notes:
```
tafsir_consulted: ibn_kathir
tafsir_influence: yes
tafsir_note: "Ibn Kathir clarifies 'الذين' refers to hypocrites based on context"
```

### 4.2 When Sources Disagree

Flag for review:
```
tafsir_consulted: [ibn_kathir, tabari]
tafsir_disagreement: yes
tafsir_note: "Ibn Kathir: believers; Tabari: general humans - flagged for adjudication"
```

---

## 5. Common Tafsir-Dependent Decisions

### 5.1 Agent Type Resolution

| Ambiguous Term | Tafsir Guidance |
|----------------|-----------------|
| الذين | Check context for specific group |
| الناس | May be believers, disbelievers, or general |
| هم | Pronoun reference from tafsir |
| قوم | Specific nation from sabab al-nuzul |

### 5.2 Behavior Form Clarification

| Ambiguous Pattern | Tafsir Helps With |
|-------------------|-------------------|
| يقولون + claim | Is it lying (speech_act) or hypocrisy (inner_state)? |
| يعملون | Physical act or general deeds? |
| في قلوبهم | Literal heart or metaphorical inner state? |

### 5.3 Evaluation Determination

| Pattern | Tafsir Clarifies |
|---------|------------------|
| Neutral description | Is it actually praise or blame in context? |
| Conditional statement | Does condition imply positive or negative? |
| Comparative statement | Which side is praised/blamed? |

---

## 6. Quality Assurance

### 6.1 Consistency Checks
- Same verse should have same tafsir interpretation across annotators
- Document any new tafsir insights for team review
- Flag verses where tafsir consultation changed initial assessment

### 6.2 Adjudication Process
When tafsir sources disagree:
1. Document all interpretations
2. Flag for lead annotator review
3. Team discussion if significant
4. Record final decision with rationale

---

## 7. Quick Reference Commands

```bash
# View tafsir for specific ayah
python tools/tafsir/tafsir_lookup.py --surah <N> --ayah <N>

# Compare multiple sources
python tools/tafsir/tafsir_lookup.py --surah <N> --ayah <N> --compare

# Search across tafsir
python tools/tafsir/tafsir_lookup.py --search "<Arabic term>"

# View range of ayat
python tools/tafsir/tafsir_lookup.py --surah <N> --ayah <start> --end-ayah <end>
```

---

## 8. Tafsir Database Status

| Source | Coverage | Status |
|--------|----------|--------|
| Ibn Kathir | 5,461/6,236 (88%) | ✅ Available |
| Al-Tabari | Pending | ⏳ To download |
| Al-Qurtubi | Pending | ⏳ To download |
| Al-Sa'di | Pending | ⏳ To download |

---

*Last Updated: Phase 4 - Tafsir Integration*
