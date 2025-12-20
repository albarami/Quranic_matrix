# QBM Annotator Quick-Start Guide
## Quranic Human-Behavior Classification Matrix

---

## 1. What You're Doing

You are classifying **human behaviors** mentioned in Quranic verses. For each ayah (verse), you will identify:

1. **Who** is doing the behavior (Agent)
2. **What** type of behavior it is (Form)
3. **How** the Quran frames it (Speech Mode, Evaluation, Deontic Signal)

---

## 2. The Five Core Fields

### Field 1: Agent Type (`agent_type`)

**Question:** Who is performing or described as having this behavior?

| Value | Arabic | When to Use |
|-------|--------|-------------|
| `AGT_BELIEVER` | المؤمن | Explicitly believers, المتقين, الذين آمنوا |
| `AGT_HYPOCRITE` | المنافق | Explicitly hypocrites, context of نفاق |
| `AGT_DISBELIEVER` | الكافر | Explicitly disbelievers, الذين كفروا |
| `AGT_HUMAN_GENERAL` | الإنسان عموماً | Humans generally, no specific group |
| `AGT_PROPHET` | النبي | Prophets, messengers |

**Example:**
- "الَّذِينَ يُؤْمِنُونَ بِالْغَيْبِ" → `AGT_BELIEVER` (explicitly those who believe)
- "خَلَقَكُم مِّن ضَعْفٍ" → `AGT_HUMAN_GENERAL` (all humans created weak)

---

### Field 2: Behavior Form (`behavior_form`)

**Question:** What category of behavior is this?

| Value | When to Use | Example |
|-------|-------------|---------|
| `speech_act` | Speaking, verbal acts | قالوا، يقولون، تقل |
| `physical_act` | Bodily actions | يصلون، ينفقون، يمشون |
| `inner_state` | Internal states, emotions, beliefs | يؤمنون، يخافون، في قلوبهم |
| `trait_disposition` | Character traits | صابرين، متقين، متواضعين |
| `relational_act` | Social interactions | إحسان بالوالدين، صلة الرحم |
| `omission` | Not doing something | لا يأكلون، لا يسجدون |
| `mixed` | Multiple types combined | يؤمنون ويقيمون الصلاة |

---

### Field 3: Speech Mode (`speech_mode`)

**Question:** How is the Quran presenting this information?

| Value | Arabic Term | Indicators |
|-------|-------------|------------|
| `command` | أمر | Imperative verbs: اعبدوا، قولوا، أنفقوا |
| `prohibition` | نهي | لا + verb: لا تقربوا، لا تقتلوا |
| `informative` | خبر | Descriptive statements about behaviors |
| `narrative` | قصص | Historical accounts |
| `parable` | مثل | Parables, similes: كمثل |

---

### Field 4: Evaluation (`evaluation`)

**Question:** Is the behavior being praised or blamed?

| Value | When to Use | Clues |
|-------|-------------|-------|
| `praise` | Behavior is commended | أولئك هم المفلحون، يحبهم الله |
| `blame` | Behavior is condemned | لهم عذاب، ساء ما يعملون |
| `warning` | Cautionary about consequences | فليحذروا، يوم تشهد |
| `promise` | Positive consequence promised | لهم جنات، أجر عظيم |
| `neutral` | No value judgment expressed | Factual descriptions |

---

### Field 5: Deontic Signal (`deontic_signal`)

**Question:** What is the normative status?

| Value | Arabic | Meaning | Typical Context |
|-------|--------|---------|-----------------|
| `amr` | أمر | Commanded/Obligatory | Imperative verbs |
| `nahy` | نهي | Prohibited/Forbidden | لا + jussive |
| `targhib` | ترغيب | Encouraged/Recommended | Praise for doing |
| `tarhib` | ترهيب | Discouraged/Warned against | Blame/warning for doing |
| `khabar` | خبر | Neutral information | Factual statements |

---

## 3. Decision Flowchart

```
START: Read the ayah
    │
    ▼
┌─────────────────────────────────┐
│ 1. IDENTIFY THE AGENT           │
│    Who is doing/having this?    │
│    Look for: الذين، هم، -ون     │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ 2. IDENTIFY BEHAVIOR FORM       │
│    Is it speech? Action?        │
│    Inner state? Relationship?   │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ 3. DETERMINE SPEECH MODE        │
│    Command? Prohibition? Info?  │
│    Look at verb form            │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ 4. ASSESS EVALUATION            │
│    Praised? Blamed? Neutral?    │
│    Look at consequences/context │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ 5. DETERMINE DEONTIC SIGNAL     │
│    Use the mapping table:       │
│    command→amr, prohibition→nahy│
│    praise→targhib, blame→tarhib │
└─────────────────────────────────┘
```

---

## 4. Common Patterns

### Pattern A: Believer Description (targhib)
```
الَّذِينَ يُؤْمِنُونَ...
├── agent_type: AGT_BELIEVER
├── behavior_form: inner_state (or mixed)
├── speech_mode: informative
├── evaluation: praise
└── deontic_signal: targhib
```

### Pattern B: Direct Command (amr)
```
وَاعْبُدُوا اللَّهَ...
├── agent_type: AGT_BELIEVER (implied)
├── behavior_form: physical_act
├── speech_mode: command
├── evaluation: neutral (the act is good, but frame is command)
└── deontic_signal: amr
```

### Pattern C: Prohibition (nahy)
```
وَلَا تَقْتُلُوا...
├── agent_type: AGT_BELIEVER (implied)
├── behavior_form: physical_act
├── speech_mode: prohibition
├── evaluation: blame (the prohibited act is bad)
└── deontic_signal: nahy
```

### Pattern D: Hypocrite Description (tarhib)
```
فِي قُلُوبِهِم مَّرَضٌ...
├── agent_type: AGT_HYPOCRITE
├── behavior_form: inner_state
├── speech_mode: informative
├── evaluation: blame
└── deontic_signal: tarhib
```

---

## 5. Edge Cases

### Heart Mentions (قلب)
When "heart" appears, determine the **semantic domain**:
- **Spiritual**: مرض القلب (hypocrisy), قلب سليم (sound heart)
- **Emotional**: رعب في قلوبهم (terror), اطمئنان (tranquility)
- **Cognitive**: لا يفقهون (don't understand), أفلا يتدبرون

### Negation
Distinguish between:
- **Prohibition**: "لا تفعل" = Don't do X (nahy)
- **Negated description**: "لا يؤمنون" = They don't believe (informative about disbelief)

### Instinctive/Automatic States
Some states are not volitional:
- "خَلَقَكُم مِّن ضَعْفٍ" (created you from weakness)
- Evaluation should be `neutral` for non-volitional states

---

## 6. Quick Reference Table

| If you see... | Likely values |
|---------------|---------------|
| الَّذِينَ آمَنُوا | agent=BELIEVER, eval=praise, deontic=targhib |
| الَّذِينَ كَفَرُوا | agent=DISBELIEVER, eval=blame, deontic=tarhib |
| المُنَافِقِينَ | agent=HYPOCRITE, eval=blame, deontic=tarhib |
| اعبدوا، قولوا، أنفقوا | mode=command, deontic=amr |
| لا تقربوا، لا تقتلوا | mode=prohibition, deontic=nahy |
| في قلوبهم | form=inner_state |
| إحسان، صلة | form=relational_act |

---

## 7. Your First 10 Practice Examples

Start with these examples from the gold standards. Annotate them WITHOUT looking at the answers, then compare:

1. **2:3** - الَّذِينَ يُؤْمِنُونَ بِالْغَيْبِ وَيُقِيمُونَ الصَّلَاةَ
2. **2:10** - فِي قُلُوبِهِم مَّرَضٌ فَزَادَهُمُ اللَّهُ مَرَضًا
3. **4:36** - وَاعْبُدُوا اللَّهَ وَلَا تُشْرِكُوا بِهِ شَيْئًا
4. **17:23** - فَلَا تَقُل لَّهُمَا أُفٍّ وَلَا تَنْهَرْهُمَا
5. **3:134** - الَّذِينَ يُنفِقُونَ فِي السَّرَّاءِ وَالضَّرَّاءِ
6. **3:134** - وَالْكَاظِمِينَ الْغَيْظَ
7. **2:177** - وَآتَى الْمَالَ عَلَىٰ حُبِّهِ ذَوِي الْقُرْبَىٰ
8. **24:24** - يَوْمَ تَشْهَدُ عَلَيْهِمْ أَلْسِنَتُهُمْ
9. **4:36** - وَبِالْوَالِدَيْنِ إِحْسَانًا
10. **30:54** - اللَّهُ الَّذِي خَلَقَكُم مِّن ضَعْفٍ

After completing these, check your answers against `docs/coding_manual/examples/gold_standard_examples.json`.

---

## 8. Getting Help

- **Full coding manual**: `docs/coding_manual/`
- **Gold standard examples**: `docs/coding_manual/examples/`
- **Controlled vocabularies**: `config/controlled_vocabularies_v1.json`

---

*Version 1.0 | Phase 2 Micro-Pilot*
