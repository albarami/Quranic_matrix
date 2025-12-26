# Graph Quality Report (Phase 6.2)

**Generated**: 2025-12-26T04:07:19.709826

---

## Summary

| Graph | Nodes | Edges | Edge Type |
|-------|-------|-------|-----------|
| Co-occurrence | 46 | 75 | CO_OCCURS_WITH |
| Semantic | 8 | 4 | CAUSES, LEADS_TO, PREVENTS... |

---

## Co-occurrence Graph (Statistical)

**Purpose**: Discovery only. NOT for causal reasoning.

### Top 20 Co-occurring Pairs

| Concept 1 | Concept 2 | Count | PMI |
|-----------|-----------|-------|-----|
| AGT_POLYTHEIST | STA_SHIRK | 18 | 1714.68 |
| AGT_PROPHET_MESSENGER | ORG_HAND | 15 | 2050.16 |
| BEH_SPI_PRAYER | AGT_PROPHET_MESSENGER | 10 | 2418.14 |
| BEH_SPI_REMEMBRANCE | AGT_PROPHET_MESSENGER | 10 | 898.16 |
| BEH_SPI_PRAYER | ORG_HAND | 9 | 2460.19 |
| BEH_SPI_REMEMBRANCE | ORG_HAND | 9 | 913.79 |
| BEH_EMO_GRATITUDE | STA_SHAKK | 8 | 3846.26 |
| BEH_SPI_REMEMBRANCE | AGT_JINN | 8 | 778.41 |
| BEH_FIN_ZAKAT | STA_NIFAQ | 7 | 1589.25 |
| BEH_SPI_REMEMBRANCE | ORG_FOOT | 7 | 583.81 |
| AGT_PROPHET_MESSENGER | ORG_EAR | 7 | 550.13 |
| ORG_EAR | ORG_FOOT | 7 | 510.83 |
| BEH_SPI_REMEMBRANCE | ORG_EAR | 6 | 350.28 |
| AGT_PROPHET_MESSENGER | ORG_FOOT | 6 | 673.62 |
| AGT_ANGEL | ORG_EAR | 6 | 533.04 |
| AGT_JINN | ORG_FOOT | 6 | 729.76 |
| BEH_SPI_TAQWA | STA_SHIRK | 5 | 773.99 |
| BEH_SPEECH_TRUTHFULNESS | AGT_BELIEVER | 5 | 671.04 |
| BEH_SPI_PRAYER | BEH_SPI_REMEMBRANCE | 5 | 898.16 |
| AGT_DISBELIEVER | STA_KUFR | 5 | 1135.18 |

---

## Semantic Graph (Typed Edges)

**Purpose**: Causal reasoning with evidence.

**Hard Rules**:
- No semantic edge without evidence offsets
- No causal chain may use co-occurrence edges

### Edge Type Distribution

- **OPPOSITE_OF**: 4 edges

### Top 20 Semantic Edges with Evidence

#### 1. BEH_SPI_SINCERITY → BEH_SPI_SHOWING_OFF (OPPOSITE_OF)

- **Confidence**: 0.9
- **Evidence count**: 3

  - Source: qurtubi, Verse: 2:41
    > "اجب من الواجبات التي يحتاج فيها إلى نية التقرب والإخلاص فلا يؤخذ عليها أجرة كالصلاة والصيام ، وقد قا..."

  - Source: qurtubi, Verse: 2:41
    > "ول في المعلمين قال درهمهم حرام وثوبهم سحت وكلامهم رياء وروى عبادة بن الصامت قال : علمت ناسا من أهل ا..."

#### 2. BEH_SOC_JUSTICE → BEH_SOC_OPPRESSION (OPPOSITE_OF)

- **Confidence**: 0.9
- **Evidence count**: 3

  - Source: tabari, Verse: 1:1
    > "رحمانا ، تسويته [ ص: 129 ] بين جميعهم جل ذكره في عدله وقضايه ، فلا يظلم احدا منهم مثقال ذره ، وان تك..."

  - Source: tabari, Verse: 1:1
    > "ص: 129 ] بين جميعهم جل ذكره في عدله وقضايه ، فلا يظلم احدا منهم مثقال ذره ، وان تك حسنه يضاعفها ويوت..."

#### 3. BEH_SPEECH_TRUTHFULNESS → BEH_SPEECH_LYING (OPPOSITE_OF)

- **Confidence**: 0.9
- **Evidence count**: 3

  - Source: qurtubi, Verse: 2:1
    > "لله تعالى ، وكان القوم في ذلك الزمان على صنفين : مصدق ، ومكذب ; فالمصدق يصدق بغير قسم ، والمكذب لا ي..."

  - Source: qurtubi, Verse: 2:1
    > "ى ، وكان القوم في ذلك الزمان على صنفين : مصدق ، ومكذب ; فالمصدق يصدق بغير قسم ، والمكذب لا يصدق مع ا..."

#### 4. STA_IMAN → STA_KUFR (OPPOSITE_OF)

- **Confidence**: 0.9
- **Evidence count**: 3

  - Source: tabari, Verse: 1:1
    > "جل الدنيا بما لطف بهم من توفيقه اياهم لطاعته ، والايمان به وبرسله ، واتباع امره واجتناب معاصيه ، مما..."

  - Source: tabari, Verse: 1:1
    > "ع امره واجتناب معاصيه ، مما خذل عنه من اشرك به ، وكفر وخالف ما امره به ، وركب معاصيه; وكان مع ذلك قد..."


---

## Quality Metrics

| Metric | Value |
|--------|-------|
| Semantic edges with evidence | 4 / 4 (100%) |
| Multi-source supported edges | 3 (75.0%) |
| Average evidence per edge | 3.0 |

---

## Validation

✅ All semantic edges have evidence offsets (char_start, char_end, quote)
✅ Co-occurrence graph is marked for discovery only
✅ Semantic graph has typed edges only
