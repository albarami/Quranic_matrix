## CODING MANUAL
## Quranic Human-Behavior Classification Matrix
دليل الترميز لمصفوفة التصنيف القرآني للسلوك البشري

**Version 1.0**  
**For Annotator Training and Reference**  
**December 2025**

---

## Table of Contents

1. Introduction and Overview  
2. Before You Begin: Setup Checklist  
3. Span Segmentation Rules  
4. Agent Identification  
5. Organ Annotation (Organic Axis)  
6. Heart Semantic Domains Playbook  
7. Situational Axis (Internal/External)  
8. Systemic Axis  
9. Spatial and Temporal Axes  
10. Action Classification  
11. Behavior Concepts Taxonomy  
12. Normative/Deontic Classification  
13. Negation Patterns  
14. Periodicity Rules  
15. Evidence and Justification  
16. Tafsir Consultation Protocol  
17. Confidence Scoring  
18. Common Mistakes to Avoid  
19. Decision Flowcharts  
Appendix A: Quick Reference Tables  
Appendix B: Worked Examples (20 Gold Standards)

---

## 1. Introduction and Overview

### 1.1 Purpose of This Manual
This coding manual provides step-by-step instructions for annotating Quranic spans with behavioral classifications. It ensures consistency across annotators and institutions.

### 1.2 Core Principles
- **NO EVIDENCE → NO LABEL**: Never assign a label without textual support
- **DIRECT vs INDIRECT**: Always track whether your evidence is explicit or inferred
- **QURAN-ONLY**: Separate Quranic signals from juristic derivations
- **CONTROLLED IDs**: Use only approved vocabulary IDs (`BEH_*`, `AGT_*`, etc.)
- **FAIL-CLOSED**: When in doubt, use `unknown` rather than guess

### 1.3 What You Will Annotate
For each behavioral span, you will annotate:
- Span boundaries (`token_start`, `token_end`)
- Agent performing the behavior
- Five axes: Organic, Situational, Systemic, Spatial, Temporal
- Action classification and textual evaluation
- Behavior concepts and thematic constructs
- Normative layer (speech mode, evaluation, deontic signal)
- Evidence type, justification, and confidence

---

## 2. Before You Begin: Setup Checklist

### 2.1 Required Materials
1. Tokenized Quran text (Uthmani Hafs, `tok_v1`)
2. Controlled vocabularies (`vocab/` in this repo)
3. This coding manual (printed or digital)
4. Access to 4 tafsir sources: Ibn Kathir, al-Tabari, al-Sa'di, al-Qurtubi
5. Annotation tool (Label Studio or equivalent)
6. Calibration examples (20 gold-standard spans)

### 2.2 Training Requirements
- Complete 8-hour training session
- Pass calibration test (≥80% agreement with gold standards)
- Review all 20 worked examples in Appendix B
- Practice with 10 additional spans under supervision

---

## 3. Span Segmentation Rules

### 3.1 Definition of a Span
A span is a contiguous, token-bounded segment within a single ayah that expresses a coherent behavioral unit.

### 3.2 Rules for Span Boundaries
**Rule 3.2.1: One Behavior Per Span**  
Each span should capture ONE primary behavior or behavioral pattern. If an ayah contains multiple distinct behaviors, create multiple spans.

**Rule 3.2.2: Include Context**  
Include sufficient tokens to capture:
- The agent (who performs the behavior)
- The action/state (what the behavior is)
- The object/target (if relevant)
- Key modifiers (how, when, where - if explicitly stated)

**Rule 3.2.3: Token Boundaries**
- `token_start`: INCLUSIVE (first token of span)
- `token_end`: EXCLUSIVE (one past last token)

Example: tokens `[0,1,2,3,4]` → `token_start=0`, `token_end=5`

**Rule 3.2.4: Multi-Ayah Behaviors**  
If a behavior spans multiple ayat:
- Create separate spans for each ayah
- Link them using `linked_spans[]` with `relation='continuation'`

### 3.3 Examples
| Ayah Text | Span Boundaries | Rationale |
|---|---|---|
| يَا أَيُّهَا الَّذِينَ آمَنُوا اتَّقُوا اللَّهَ | tokens 0-5 | Complete command with agent and action |
| وَالَّذِينَ يُنفِقُونَ أَمْوَالَهُمْ... | tokens 0-4 | Agent + spending behavior |
| يَوْمَ تَشْهَدُ عَلَيْهِمْ أَلْسِنَتُهُمْ | tokens 0-4 | Testimony behavior with organ |

⚠️ NEVER split a grammatically connected phrase. Keep subject-verb-object together.

---

## 4. Agent Identification

### 4.1 Agent Types
| ID | When to Use | Quranic Indicators |
|---|---|---|
| AGT_BELIEVER | Explicitly identified believers | الذين آمنوا، المؤمنين |
| AGT_HYPOCRITE | Explicitly identified hypocrites | المنافقين، الذين في قلوبهم مرض |
| AGT_DISBELIEVER | Explicitly identified disbelievers | الذين كفروا، الكافرين |
| AGT_HUMAN_GENERAL | Generic human reference | الإنسان، الناس، بني آدم |
| AGT_PROPHET_MESSENGER | Prophets/messengers | النبي، الرسول، named prophets |
| AGT_COLLECTIVE | Groups/nations | قوم، أمة، حزب |
| AGT_UNKNOWN | Cannot determine from text | Pronoun without clear referent |

### 4.2 Explicit vs Implicit Agents
- EXPLICIT: Agent is named or clearly identified in the span
- IMPLICIT: Agent is inferred from context, pronouns, or verb conjugation

Always set `agent.explicit = true/false` accordingly.

### 4.3 Pronoun Resolution
When the agent is a pronoun (هم، هو، أنتم، etc.):
1. Check immediate context (same ayah) for referent
2. Check preceding ayat for referent
3. Consult tafsir if unclear
4. Set `justification_code = JST_PRONOUN_RESOLVED`
5. If still unclear, use `AGT_UNKNOWN`

---

## 5. Organ Annotation (Organic Axis)

### 5.1 When to Annotate Organs
Annotate organs ONLY when:
- The organ is explicitly mentioned in the span
- The organ is functioning as: tool, perception, witness, or metaphor

### 5.2 Organ Labels (common)
| ID | Arabic Terms | Common Contexts |
|---|---|---|
| ORG_HEART | قلب، قلوب، فؤاد | Faith, understanding, emotion, sincerity |
| ORG_TONGUE | لسان، ألسنة | Speech, testimony, lying/truth |
| ORG_EYE | عين، أعين، بصر، أبصار | Seeing, lowering gaze, blindness |
| ORG_EAR | أذن، آذان، سمع | Hearing, listening, deafness |
| ORG_HAND | يد، أيدي | Action, giving, harm |
| ORG_FOOT | رجل، أرجل، قدم | Walking, standing, movement |
| ORG_SKIN | جلد، جلود | Covering, witness on Day of Judgment |
| ORG_CHEST | صدر، صدور | Emotion, tightness, expansion |
| ORG_FACE | وجه، وجوه | Direction, expression, honor |

### 5.3 Organ Roles
| Role | When to Use | Example |
|---|---|---|
| ROLE_TOOL | Organ performs action | ألسنتهم تصف الكذب |
| ROLE_PERCEPTION | Organ perceives | لهم أعين لا يبصرون بها |
| ROLE_ACCOUNTABILITY_WITNESS | Organ testifies | تشهد عليهم ألسنتهم |
| ROLE_METAPHOR | Non-literal usage | في قلوبهم مرض |

💡 When ORG_HEART is annotated, you MUST also annotate `organ_semantic_domains`. See Section 6.

---

## 6. Heart Semantic Domains Playbook

### 6.1 Why Special Treatment for Heart?
The Quranic 'qalb' (قلب) carries multiple semantic domains that rarely map to the English 'heart'. You must identify which domain(s) are active.

### 6.2 Domain Identification
| Domain | Quranic Indicators | Example Contexts |
|---|---|---|
| DOM_PHYSIOLOGICAL | Rare - only when literal physical heart | Extremely rare in behavioral contexts |
| DOM_COGNITIVE | يعقلون بها، يفقهون، يعلمون + قلب | Understanding, reasoning, comprehension |
| DOM_SPIRITUAL | إيمان، كفر، نفاق، خشوع، إخلاص + قلب | Faith, worship orientation, sincerity |
| DOM_EMOTIONAL | خوف، حزن، فرح، طمأنينة، قسوة + قلب | Fear, joy, tranquility, hardness |

### 6.3 Decision Flowchart for Heart Domains
1. Is the heart described with faith/disbelief/hypocrisy terms? → SPIRITUAL
2. Is the heart described with understanding/reasoning terms? → COGNITIVE
3. Is the heart described with emotion terms (fear, joy, hardness)? → EMOTIONAL
4. Is the heart literally the physical organ? → PHYSIOLOGICAL (rare)
5. Multiple apply? → Select all that apply, mark `primary_organ_semantic_domain`

### 6.4 Examples
- أُولَٰئِكَ كَتَبَ فِي قُلُوبِهِمُ الْإِيمَانَ → DOM_SPIRITUAL
- لَهُمْ قُلُوبٌ لَّا يَفْقَهُونَ بِهَا → DOM_COGNITIVE
- وَتَطْمَئِنُّ قُلُوبُهُم بِذِكْرِ اللَّهِ → DOM_EMOTIONAL + DOM_SPIRITUAL

---

## 7. Situational Axis (Internal/External)

### 7.1 Core Distinction
| Category | Definition | Examples |
|---|---|---|
| EXTERNAL (ظاهر) | Observable speech or bodily acts | Speaking, walking, spending, praying (physical) |
| INTERNAL (باطن) | Beliefs, intentions, emotions, cognition | Believing, intending, fearing, understanding |
| MIXED | Both internal and external elements | Prayer with sincerity, spending ostentatiously |

### 7.2 Decision Rules
- If the behavior is ONLY observable action → external
- If the behavior is ONLY mental/spiritual state → internal
- If both are explicitly present → mixed
- If unclear → unknown

### 7.3 Optional Domain Tags
When internal, you may add domain tags:
- emotional: fear, hope, love, anger, gratitude
- spiritual: faith, worship, sincerity, repentance
- cognitive: understanding, knowledge, reflection, doubt
- psychological: arrogance, humility, patience
- social: (rarely internal, but possible for attitudes toward others)

---

## 8. Systemic Axis

### 8.1 Multi-Label Selection
Select ALL that apply:
| ID | When to Select | Example Behaviors |
|---|---|---|
| SYS_SELF | Behavior affects self | Self-purification, self-harm, patience |
| SYS_CREATION | Behavior involves other creatures | Kindness to parents, oppression, charity |
| SYS_GOD | Behavior oriented toward Allah | Prayer, worship, gratitude, shirk |
| SYS_COSMOS | Behavior involves nature/universe | Reflection on creation, stewardship |
| SYS_LIFE | Behavior about worldly/hereafter life | Seeking dunya vs akhira |

### 8.2 Primary Systemic
When multiple apply, identify the PRIMARY one based on:
1. What is the MAIN orientation of the behavior?
2. What is EXPLICITLY stated vs inferred?
3. Tie-breaker: SYS_GOD > SYS_CREATION > SYS_SELF > SYS_LIFE > SYS_COSMOS

---

## 9. Spatial and Temporal Axes

### 9.1 Spatial Axis
Annotate ONLY when location is explicitly mentioned or clearly implied:
| ID | Indicators |
|---|---|
| LOC_HOME | بيوتكم، في بيوتهن، مساكن |
| LOC_MASJID | المسجد، المساجد، بيوت الله |
| LOC_MARKET | الأسواق، التجارة |
| LOC_BATTLEFIELD | سبيل الله (context), الحرب |
| LOC_SACRED | الحرم، مكة، البيت الحرام |
| LOC_UNKNOWN | No spatial reference (DEFAULT) |

### 9.2 Temporal Axis
Annotate ONLY when time is explicitly mentioned:
| ID | Indicators |
|---|---|
| TMP_FAJR | الفجر، صلاة الفجر |
| TMP_NIGHT | الليل، قيام الليل، بالأسحار |
| TMP_FRIDAY | يوم الجمعة |
| TMP_RAMADAN | شهر رمضان |
| TMP_UNKNOWN | No temporal reference (DEFAULT) |

⚠️ Do NOT infer spatial/temporal from general knowledge. Only annotate what the TEXT states.

---

## 10. Action Classification

### 10.1 Action Class (AX_ACTION_CLASS)
| ID | Definition | When to Use |
|---|---|---|
| ACT_INSTINCTIVE_OR_AUTOMATIC | Involuntary, biological, reflexive | Hunger, thirst, death, weakness, biological states |
| ACT_VOLITIONAL | Deliberate, chosen, acquired | Prayer, lying, spending, oppression |
| ACT_UNKNOWN | Cannot determine | Most spans (DEFAULT if unclear) |

⚠️ ACT_INSTINCTIVE is RARE. Only use with high tafsir agreement for involuntary states.

### 10.2 Action Textual Evaluation (AX_ACTION_TEXTUAL_EVAL)
| ID | When to Use | Textual Cues |
|---|---|---|
| EVAL_SALIH | Text EXPLICITLY praises or rewards | مدح صريح، وعد بالجنة |
| EVAL_GHAYR_SALIH | Text EXPLICITLY blames or threatens | ذم صريح، وعيد بالنار |
| EVAL_NEUTRAL | No evaluation in text | Pure narration without judgment |
| EVAL_NOT_APPLICABLE | Instinctive acts | Involuntary states not subject to moral eval |
| EVAL_UNKNOWN | Cannot determine (DEFAULT) | Ambiguous evaluation |

### 10.3 Rule: Instinctive → Not Applicable
IF `action_class = ACT_INSTINCTIVE_OR_AUTOMATIC`  
THEN `action_textual_eval = EVAL_NOT_APPLICABLE` (unless text explicitly evaluates)

---

## 11. Behavior Concepts Taxonomy

### 11.1 Categories
Behavior concepts are organized into categories:
- Speech: `BEH_SPEECH_*`
- Financial: `BEH_FIN_*`
- Emotional: `BEH_EMO_*`
- Spiritual: `BEH_SPI_*`
- Social: `BEH_SOC_*`
- Cognitive: `BEH_COG_*`
- Physical: `BEH_PHY_*`

### 11.2 Selection Rules
1. Select the MOST SPECIFIC concept that fits
2. Maximum 3 concepts per span (if truly multiple behaviors)
3. Each concept MUST have a corresponding assertion in `assertions[]`
4. If no concept fits, leave `behavior.concepts` empty

### 11.3 Common Concepts for Pilot
- BEH_SPEECH_TRUTHFULNESS | BEH_SPEECH_LYING
- BEH_FIN_CHARITY | BEH_FIN_SPENDING | BEH_FIN_USURY
- BEH_EMO_GRATITUDE | BEH_EMO_PATIENCE | BEH_EMO_FEAR_ALLAH
- BEH_SPI_FAITH | BEH_SPI_SINCERITY | BEH_SPI_HYPOCRISY | BEH_SPI_PRAYER
- BEH_SOC_JUSTICE | BEH_SOC_OPPRESSION | BEH_SOC_KINDNESS_PARENTS
- BEH_COG_REFLECTION | BEH_COG_ARROGANCE | BEH_COG_KNOWLEDGE

---

## 12. Normative/Deontic Classification

### 12.1 Speech Mode
| Mode | Indicators | Example |
|---|---|---|
| command | Imperative verb (افعل، افعلوا) | أقيموا الصلاة |
| prohibition | لا + jussive (لا تفعل) | لا تقربوا الزنا |
| informative | Declarative statement | الله يعلم ما في قلوبكم |
| narrative | Story/historical account | إذ قال موسى لقومه |
| parable | مثل، كمثل patterns | مثلهم كمثل... |

### 12.2 Evaluation
- praise: Explicit commendation
- blame: Explicit condemnation
- warning: Threat of consequence
- promise: Promise of reward
- neutral: No evaluative language

### 12.3 Deontic Signal
| Signal | Grammatical Form | Notes |
|---|---|---|
| amr | Imperative | Command - do this |
| nahy | لا + jussive | Prohibition - don't do this |
| targhib | Praise + reward promise | Encouragement |
| tarhib | Blame + punishment threat | Warning/discouragement |
| khabar | Pure report | No deontic force |

### 12.4 Decision Flowchart
1. Is there an imperative verb? → amr
2. Is there لا + jussive? → nahy
3. Is there praise + reward? → targhib
4. Is there blame + threat? → tarhib
5. Otherwise → khabar

---

## 13. Negation Patterns

### 13.1 The 'لا…ولكن' Pattern
لا X ولكن Y / ليس X ولكن Y / ما X ولكن Y

### 13.2 Annotation Rule: Create TWO Assertions
When you encounter this pattern within one span:
- Assertion 1 (X - negated part):
  - `negated: true`
  - `negation_type: absolute` OR `conditional`
  - `justification_code: JST_GRAMMATICAL_FORM`
- Assertion 2 (Y - affirmed part):
  - `negated: false`
  - justification references the contrast with X

### 13.3 Negation Types
| Type | When to Use | Example |
|---|---|---|
| absolute | Complete negation | لا... |
| conditional | Negation with condition/exception | ... إلا ... |
| exceptionless_affirmation | Double negation = affirmation | ما أنا إلا ... |

---

## 14. Periodicity Rules

### 14.1 When to Annotate Periodicity
ONLY annotate when the text provides EXPLICIT grammatical or lexical support:
| Category | Grammatical Indicators | Confidence |
|---|---|---|
| PER_HABIT | كان + imperfect | High if كان pattern |
| PER_ROUTINE_DAILY | Explicit 'كل يوم' | Only with explicit marker |
| PER_AUTOMATIC | Involuntary states | Rare - needs tafsir support |
| PER_UNKNOWN | No markers (DEFAULT) | Always safe default |

### 14.2 Grammatical Indicators
- GRM_KANA_IMPERFECT: كان/كانوا + مضارع
- GRM_NOMINAL_SENTENCE_STABILITY: هم المؤمنون
- GRM_EXPLICIT_FREQUENCY_TERM: كل يوم، دائماً، أبداً
- GRM_IMPERFECT_GENERAL: يفعلون (without كان)

⚠️ Do NOT import periodicity from Sunnah/fiqh. Store external info in `frequency_prescription` only.

---

## 15. Evidence and Justification

### 15.1 Support Type
| Type | Definition | Confidence Impact |
|---|---|---|
| direct | Explicit wording/structure | Higher confidence allowed |
| indirect | Inferred (context, metaphor, tafsir) | Lower confidence cap |

### 15.2 Indication Tags
- dalalah_mantuq
- dalalah_mafhum
- narrative_inference
- metaphor_metonymy
- sabab_nuzul_used

### 15.3 Justification Codes
- JST_EXPLICIT_MENTION
- JST_GRAMMATICAL_FORM
- JST_TAFSIR_CONSENSUS
- JST_NARRATIVE_CONTEXT
- JST_PRONOUN_RESOLVED
- JST_CONTEXT_ONLY
- JST_METAPHOR_RESOLVED

---

## 16. Tafsir Consultation Protocol

### 16.1 When to Consult Tafsir
- ALWAYS for indirect evidence
- When pronouns need resolution
- When metaphor/metonymy is present
- When meaning is ambiguous
- For disputed or sensitive passages

### 16.2 Required Sources
Consult at least 2 of:
- Ibn Kathir
- al-Tabari
- al-Sa'di
- al-Qurtubi

### 16.3 Agreement Levels
| Level | Definition | Action |
|---|---|---|
| high | All sources agree | Proceed with confidence |
| mixed | Differ on secondary | Use primary meaning, note variance |
| disputed | Disagreement affects label | Set label to unknown or lower confidence |

### 16.4 Recording Tafsir Consultation
- `tafsir.sources_used`: `['IbnKathir', 'Tabari']`
- `tafsir.agreement_level`: `high`
- `tafsir.notes`: short summary

---

## 17. Confidence Scoring

### 17.1 Evidence Strength Score (ESS) Components
| Component | Range | Guidance |
|---|---:|---|
| directness | 0.0-1.0 | 1.0=explicit |
| clarity | 0.0-1.0 | Meaning clarity |
| specificity | 0.0-1.0 | Label match precision |
| tafsir_agreement | 0.0-1.0 | 1.0=high |

### 17.2 Final ESS Calculation
`final = directness × clarity × specificity × tafsir_agreement`

### 17.3 Confidence Caps
- If indirect and clarity < 0.7 → confidence ≤ 0.75
- If metaphor_metonymy used → confidence ≤ 0.80 (unless tafsir resolves)
- If tafsir disputed → confidence ≤ 0.70 OR set to unknown

### 17.4 Confidence Scale
| Score | Meaning | When to Use |
|---|---|---|
| 0.90-1.00 | Very High | Explicit, clear, tafsir consensus |
| 0.75-0.89 | High | Strong indirect with tafsir support |
| 0.60-0.74 | Moderate | Some ambiguity |
| 0.40-0.59 | Low | Significant uncertainty |
| < 0.40 | Very Low | Consider unknown instead |

---

## 18. Common Mistakes to Avoid

### 18.1 Labeling Without Evidence
WRONG: Assigning BEH_SPI_SINCERITY just because behavior seems good.  
RIGHT: Only assign if text explicitly indicates sincerity (إخلاص، خالص).

### 18.2 Importing External Knowledge
WRONG: Setting PER_ROUTINE_DAILY for salah because we know 5x daily.  
RIGHT: Use `frequency_prescription` for Sunnah-derived info.

### 18.3 Confusing Juristic and Textual
WRONG: EVAL_GHAYR_SALIH because fiqh says haram.  
RIGHT: Only mark if the TEXT blames/threatens.

### 18.4 Over-Annotating Organs
WRONG: Annotating ORG_HAND for any “doing”.  
RIGHT: Only annotate when يد/أيدي explicitly appears.

### 18.5 Single Domain for Heart
WRONG: Always selecting DOM_SPIRITUAL.  
RIGHT: Distinguish cognitive vs emotional vs spiritual from context.

### 18.6 Guessing Agent
WRONG: Assuming الذين means believers.  
RIGHT: Use context; AGT_UNKNOWN if unclear.

### 18.7 Splitting Grammatical Units
WRONG: Splitting subject from verb.  
RIGHT: Keep grammatically connected phrases together.

---

## 19. Decision Flowcharts

### 19.1 Master Annotation Flowchart
1. Read the ayah in full context
2. Identify behavioral span boundaries
3. Identify the agent
4. Classify behavior form
5. Annotate organs if present
6. Mark situational (internal/external)
7. Mark systemic
8. Mark spatial/temporal if explicit
9. Classify action
10. Evaluate textually
11. Select behavior concepts
12. Mark normative layer
13. Check for negation patterns
14. Assess periodicity
15. Determine evidence type and justification
16. Consult tafsir if needed
17. Calculate confidence
18. Review and submit

### 19.2 Organ Annotation Flowchart
1. Is an organ word present? If NO → skip
2. Which organ? → select from ORG_*
3. Is it ORG_HEART? If YES → Heart Domains Playbook
4. What role? → ROLE_*
5. Set support_type (usually direct for explicit organs)
6. Set confidence (usually high for explicit mentions)

---

## Appendix A: Quick Reference Tables

### A.1 Controlled ID Prefixes
| Prefix | Category | Example |
|---|---|---|
| AX_ | Axis | AX_ORGAN |
| AGT_ | Agent | AGT_BELIEVER |
| ORG_ | Organ | ORG_HEART |
| SYS_ | Systemic | SYS_GOD |
| LOC_ | Location | LOC_HOME |
| TMP_ | Temporal | TMP_NIGHT |
| BEH_ | Behavior | BEH_SPI_FAITH |
| THM_ | Thematic | THM_REWARD |
| ACT_ | Action Class | ACT_VOLITIONAL |
| EVAL_ | Action Eval | EVAL_SALIH |
| PER_ | Periodicity | PER_HABIT |
| GRM_ | Grammatical | GRM_KANA_IMPERFECT |
| JST_ | Justification | JST_EXPLICIT_MENTION |

### A.2 Default Values
| Field | Default | When to Change |
|---|---|---|
| periodicity.category | PER_UNKNOWN | Only with explicit support |
| spatial | LOC_UNKNOWN | Only when explicitly stated |
| temporal | TMP_UNKNOWN | Only when explicitly stated |
| action_class | ACT_UNKNOWN | When clearly instinctive/volitional |
| action_textual_eval | EVAL_UNKNOWN | When text explicitly evaluates |

---

## Appendix B: Worked Examples (20 Gold Standards)
This appendix contains 20 fully annotated examples for calibration. See the companion file:
- `docs/coding_manual/examples/gold_standard_examples.json`

---

## Document Control
| Version | Date | Changes |
|---|---|---|
| 1.0 | December 2025 | Initial release |


