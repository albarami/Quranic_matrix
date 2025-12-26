# Graph Quality Report (Phase 6.3)

**Generated**: 2025-12-26T08:02:02.298058

---

## Summary

| Graph | Total Nodes | Isolated Nodes | Edges |
|-------|-------------|----------------|-------|
| Co-occurrence | 126 | 12 | 4879 |
| Semantic | 126 | 31 | 1330 |

---

## Node Counts by Type

| Type | Count |
|------|-------|
| BEHAVIOR | 73 |
| AGENT | 14 |
| ORGAN | 11 |
| HEART_STATE | 12 |
| CONSEQUENCE | 16 |

---

## Co-occurrence Graph (Statistical)

**Purpose**: Discovery only. NOT for causal reasoning.

### Top 20 Co-occurring Pairs

| Concept 1 | Concept 2 | Count | PMI |
|-----------|-----------|-------|-----|
| AGT_ALLAH | ORG_HAND | 42490 | 1.06 |
| AGT_ALLAH | CSQ_DHULL | 40396 | 1.07 |
| AGT_ALLAH | BEH_SPI_REMEMBRANCE | 25996 | 1.07 |
| ORG_HAND | CSQ_DHULL | 25528 | 1.15 |
| AGT_ALLAH | BEH_COG_KNOWLEDGE | 18470 | 1.07 |
| BEH_SPI_REMEMBRANCE | CSQ_DHULL | 18332 | 1.36 |
| ORG_HAND | BEH_SPI_REMEMBRANCE | 17010 | 1.20 |
| BEH_COG_KNOWLEDGE | CSQ_DHULL | 11842 | 1.24 |
| AGT_ALLAH | AGT_JINN | 11729 | 1.04 |
| AGT_ALLAH | ORG_SOUL | 11156 | 1.07 |
| BEH_COG_KNOWLEDGE | ORG_HAND | 11035 | 1.09 |
| AGT_ALLAH | ORG_FOOT | 10709 | 1.08 |
| AGT_ALLAH | BEH_SPI_HAJJ | 10308 | 1.07 |
| AGT_ALLAH | ORG_FACE | 8357 | 1.07 |
| ORG_HAND | ORG_FOOT | 7736 | 1.34 |
| AGT_ALLAH | ORG_EYE | 7603 | 1.06 |
| ORG_HAND | AGT_JINN | 7590 | 1.16 |
| AGT_ALLAH | BEH_SPI_DISBELIEF | 7504 | 1.04 |
| ORG_SOUL | CSQ_DHULL | 6959 | 1.20 |
| BEH_COG_KNOWLEDGE | BEH_SPI_REMEMBRANCE | 6836 | 1.11 |

---

## Semantic Graph (Typed Edges)

**Purpose**: Causal reasoning with evidence.

**Hard Rules**:
- No semantic edge without evidence offsets
- No causal chain may use co-occurrence edges

### Edge Type Distribution

- **STRENGTHENS**: 291 edges
- **OPPOSITE_OF**: 367 edges
- **CAUSES**: 575 edges
- **PREVENTS**: 97 edges

### Top 20 Semantic Edges with Evidence

#### 1. AGT_ALLAH → AGT_ANGEL (STRENGTHENS)

- **Confidence**: 0.95
- **Evidence count**: 5

  - Source: ibn_kathir, Verse: 1:1
    > "حدثنا عبد الله بن عمر بن ابان الكوفي ، حدثنا علي بن هشام بن البريد عن يزيد بن زياد ، عن عبد الملك بن..."

  - Source: ibn_kathir, Verse: 1:4
    > "ب : انه بلغه ان رسول الله صلي الله عليه وسلم وابا بكر وعمر وعثمان ومعاويه وابنه يزيد بن معاويه كانوا..."

#### 2. AGT_ALLAH → AGT_HUMAN_GENERAL (OPPOSITE_OF)

- **Confidence**: 0.8
- **Evidence count**: 2

  - Source: ibn_kathir, Verse: 1:1
    > "قدر علي منعه ودفعه الا الله الذي خلقه ، ولا يقبل مصانعه ، ولا يداري بالاحسان ، بخلاف العدو من نوع ال..."

  - Source: baghawi, Verse: 2:8
    > "صحابهم حيث أظهروا كلمة الإسلام ليَسْلَمُوا من النبي صلى الله عليه وسلم وأصحابه واعتقدوا خلافها، وأكث..."

#### 3. BEH_FIN_HOARDING → AGT_HUMAN_GENERAL (OPPOSITE_OF)

- **Confidence**: 0.7000000000000001
- **Evidence count**: 2

  - Source: ibn_kathir, Verse: 1:1
    > "قدر علي منعه ودفعه الا الله الذي خلقه ، ولا يقبل مصانعه ، ولا يداري بالاحسان ، بخلاف العدو من نوع ال..."

  - Source: ibn_kathir, Verse: 7:11
    > "ولكن لما كان ذلك منه علي الاباء الذين هم اصل صار كانه واقع علي الابناء . وهذا بخلاف قوله تعالي : ( و..."

#### 4. AGT_ALLAH → BEH_COG_KNOWLEDGE (OPPOSITE_OF)

- **Confidence**: 0.95
- **Evidence count**: 5

  - Source: tabari, Verse: 1:1
    > "الذي امروا به من تسميته عند افتتاح تلاوه تنزيل الله ، وصدور رسايلهم وكتبهم .ولا خلاف بين الجميع من ع..."

  - Source: qurtubi, Verse: 1:2
    > "قصود إصابة المعنى . قال ابن المنذر : لا يجزئه ذلك ; لأنه خلاف ما أمر الله به ، وخلاف ما علم النبي صل..."

#### 5. AGT_ALLAH → ORG_FOOT (OPPOSITE_OF)

- **Confidence**: 0.95
- **Evidence count**: 5

  - Source: ibn_kathir, Verse: 1:2
    > "ا حرره بعض المتاخرين ، والله اعلم .وقال ابو نصر اسماعيل بن حماد الجوهري : الحمد نقيض الذم ، تقول : ح..."

  - Source: baghawi, Verse: 2:265
    > "بالثواب وتصديق بوعد الله ويعلمون أن ما أخرجوا خير لهم مما تركوا وقيل على يقين بإخلاف الله عليهم .وقا..."

#### 6. BEH_COG_KNOWLEDGE → ORG_FOOT (OPPOSITE_OF)

- **Confidence**: 0.95
- **Evidence count**: 3

  - Source: ibn_kathir, Verse: 1:2
    > "ا حرره بعض المتاخرين ، والله اعلم .وقال ابو نصر اسماعيل بن حماد الجوهري : الحمد نقيض الذم ، تقول : ح..."

  - Source: qurtubi, Verse: 2:184
    > "أطعم عن الماضي ، فإذا أفطر قضاه . إسناد صحيح . قال علماؤنا : وأقوال الصحابة على خلاف القياس قد يحتج ..."

#### 7. BEH_SPI_REPENTANCE → BEH_COG_KNOWLEDGE (OPPOSITE_OF)

- **Confidence**: 0.7000000000000001
- **Evidence count**: 2

  - Source: qurtubi, Verse: 1:2
    > "سورة الأعراف ، والأنفال ، والتوبة ، ونحوها .( الثالث ) : فاتحة الكتاب ، من غير خلاف بين العلماء ; وس..."

  - Source: qurtubi, Verse: 4:17
    > "وكثرة الاستغفار ، وقد تقدم في " آل عمران " كثير من معاني التوبة وأحكامها . ولا خلاف فيما أعلمه أن ال..."

#### 8. BEH_SPI_HAJJ → BEH_SPI_PRAYER (OPPOSITE_OF)

- **Confidence**: 0.7000000000000001
- **Evidence count**: 2

  - Source: qurtubi, Verse: 1:2
    > "وله تعالى : ولقد آتيناك سبعا من المثاني والقرآن العظيم والحجر مكية بإجماع . ولا خلاف أن فرض الصلاة ك..."

  - Source: qurtubi, Verse: 22:29
    > "، وذلك الشيء واجب في الحج قد جاز وقته ، فإن تطوعه ذلك يصير للواجب لا للتطوع ؛ بخلاف الصلاة . فإذا كا..."

#### 9. BEH_FIN_HOARDING → AGT_ALLAH (OPPOSITE_OF)

- **Confidence**: 0.95
- **Evidence count**: 5

  - Source: qurtubi, Verse: 1:2
    > "( ما لي أنازع القرآن ) لما أفتى بخلافه ، وقول الزهري في حديث ابن أكيمة : فانتهى الناس عن القراءة مع ..."

  - Source: qurtubi, Verse: 2:4
    > "ا .قوله تعالى : بما أنزل إليك يعني القرآنوما أنزل من قبلك يعني الكتب السالفة ، بخلاف ما فعله اليهود ..."

#### 10. AGT_ALLAH → BEH_COG_UNDERSTANDING (OPPOSITE_OF)

- **Confidence**: 0.95
- **Evidence count**: 5

  - Source: qurtubi, Verse: 1:2
    > "قصود إصابة المعنى . قال ابن المنذر : لا يجزئه ذلك ; لأنه خلاف ما أمر الله به ، وخلاف ما علم النبي صل..."

  - Source: qurtubi, Verse: 1:2
    > "يجزئه ذلك ; لأنه خلاف ما أمر الله به ، وخلاف ما علم النبي صلى الله عليه وسلم ، وخلاف جماعات المسلمين..."

#### 11. BEH_COG_KNOWLEDGE → BEH_COG_UNDERSTANDING (OPPOSITE_OF)

- **Confidence**: 0.95
- **Evidence count**: 5

  - Source: qurtubi, Verse: 1:2
    > "يجزئه ذلك ; لأنه خلاف ما أمر الله به ، وخلاف ما علم النبي صلى الله عليه وسلم ، وخلاف جماعات المسلمين..."

  - Source: qurtubi, Verse: 2:282
    > "ا ؛ لأن سعره واحد ، والله أعلم . وأما الشرط الخامس وهو أن يكون الأجل معلوما فلا خلاف ، فيه بين الأمة..."

#### 12. ORG_EYE → AGT_ALLAH (OPPOSITE_OF)

- **Confidence**: 0.95
- **Evidence count**: 5

  - Source: tabari, Verse: 1:4
    > "ه .ومره : هو ابن شراحيل الهمداني الكوفي ، وهو تابعي ثقه ، من كبار التابعين ، ليس فيه خلاف بينهم .وال..."

  - Source: tabari, Verse: 2:51
    > "يوم تمام يومين، وتمام اربعين.قال ابو جعفر: وذلك خلاف ما جاءت به الروايه عن اهل التاويل، وخلاف ظاهر ا..."

#### 13. AGT_ALLAH → BEH_SPI_REMEMBRANCE (STRENGTHENS)

- **Confidence**: 0.95
- **Evidence count**: 5

  - Source: tabari, Verse: 1:7
    > "صديق. والكتاب الذي انزل علي نبينا محمد صلي الله عليه وسلم يحوي معاني ذلك كله , ويزيد عليه كثيرا من ا..."

  - Source: tabari, Verse: 2:55
    > "ابن زيد: (حتي نري الله جهره)، حتي يطلع الينا.950 - حدثنا بشر بن معاذ قال، حدثنا يزيد قال، حدثنا سعيد..."

#### 14. BEH_SPI_HYPOCRISY → AGT_ALLAH (STRENGTHENS)

- **Confidence**: 0.8
- **Evidence count**: 2

  - Source: ibn_kathir, Verse: 2:1
    > "نفاق حتي يصبح ، قال : فكان يقروهما كل يوم وليله سوي جزيه .[ قال ايضا : ] وحدثنا يزيد ، عن ورقاء بن ا..."

  - Source: tabari, Verse: 2:77
    > "ما جاء به، نفاقا وخداعا لله ولرسوله وللمومنين؟ كما:-1350 - حدثنا بشر قال، حدثنا يزيد قال، حدثنا سعيد..."

#### 15. ORG_FACE → BEH_COG_DOUBT (OPPOSITE_OF)

- **Confidence**: 0.7000000000000001
- **Evidence count**: 2

  - Source: saadi, Verse: 2:2
    > "م العظيم, والحق المبين. فـ { لَا رَيْبَ فِيهِ } ولا شك بوجه من الوجوه، ونفي الريب عنه, يستلزم ضده, ا..."

  - Source: saadi, Verse: 2:2
    > ", والحق المبين. فـ { لَا رَيْبَ فِيهِ } ولا شك بوجه من الوجوه، ونفي الريب عنه, يستلزم ضده, اذ ضد الر..."

#### 16. ORG_FACE → BEH_COG_KNOWLEDGE (OPPOSITE_OF)

- **Confidence**: 0.95
- **Evidence count**: 5

  - Source: saadi, Verse: 2:2
    > "م العظيم, والحق المبين. فـ { لَا رَيْبَ فِيهِ } ولا شك بوجه من الوجوه، ونفي الريب عنه, يستلزم ضده, ا..."

  - Source: saadi, Verse: 2:2
    > ", والحق المبين. فـ { لَا رَيْبَ فِيهِ } ولا شك بوجه من الوجوه، ونفي الريب عنه, يستلزم ضده, اذ ضد الر..."

#### 17. ORG_FACE → BEH_COG_CERTAINTY (OPPOSITE_OF)

- **Confidence**: 0.9000000000000001
- **Evidence count**: 3

  - Source: saadi, Verse: 2:2
    > "م العظيم, والحق المبين. فـ { لَا رَيْبَ فِيهِ } ولا شك بوجه من الوجوه، ونفي الريب عنه, يستلزم ضده, ا..."

  - Source: saadi, Verse: 2:2
    > ", والحق المبين. فـ { لَا رَيْبَ فِيهِ } ولا شك بوجه من الوجوه، ونفي الريب عنه, يستلزم ضده, اذ ضد الر..."

#### 18. AGT_ALLAH → BEH_COG_ARROGANCE (CAUSES)

- **Confidence**: 0.95
- **Evidence count**: 5

  - Source: saadi, Verse: 2:2
    > "ولم يقبلوا هدي الله, فقامت عليهم به الحجه, ولم ينتفعوا به لشقايهم، واما المتقون الذين اتوا بالسبب ال..."

  - Source: saadi, Verse: 2:151
    > "} لانهم كانوا قبل بعثته, في ضلال مبين, لا علم ولا عمل، فكل علم او عمل, نالته هذه الامه فعلي يده صلي ..."

#### 19. AGT_ALLAH → CSQ_HIDAYA (CAUSES)

- **Confidence**: 0.95
- **Evidence count**: 5

  - Source: saadi, Verse: 2:2
    > "ولم يقبلوا هدي الله, فقامت عليهم به الحجه, ولم ينتفعوا به لشقايهم، واما المتقون الذين اتوا بالسبب ال..."

  - Source: jalalayn, Verse: 2:211
    > "من والسلوى فبدَّلوها كفرا «ومن يبدِّل نعمه الله» أي ما أنعم به عليه من الآيات لأنها سبب الهداية «من ..."

#### 20. AGT_ALLAH → BEH_SPI_TAQWA (CAUSES)

- **Confidence**: 0.95
- **Evidence count**: 5

  - Source: saadi, Verse: 2:2
    > "ولم يقبلوا هدي الله, فقامت عليهم به الحجه, ولم ينتفعوا به لشقايهم، واما المتقون الذين اتوا بالسبب ال..."

  - Source: saadi, Verse: 2:187
    > "علم تحريمه لم يفعله، فاذا بين الله للناس اياته، لم يبق لهم عذر ولا حجه، فكان ذلك سببا للتقو..."


---

## Quality Metrics

| Metric | Value |
|--------|-------|
| Semantic edges with evidence | 1330 / 1330 (100%) |
| Multi-source supported edges | 777 (58.4%) |
| Average evidence per edge | 3.2 |

---

## Validation

✅ All semantic edges have evidence offsets (char_start, char_end, quote)
✅ Co-occurrence graph is marked for discovery only
✅ Semantic graph has typed edges only
