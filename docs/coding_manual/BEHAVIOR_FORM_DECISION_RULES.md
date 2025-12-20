# Behavior Form Decision Rules

## Purpose
These rules clarify edge cases to improve inter-annotator agreement **legitimately** - by providing clearer definitions, not by forcing arbitrary choices.

## Core Definitions

| Form | Definition | Key Indicator |
|------|------------|---------------|
| **inner_state** | Mental/emotional states, beliefs, intentions | Occurs inside the person |
| **speech_act** | Verbal communication as the primary behavior | Speaking/saying is the action |
| **physical_act** | Bodily actions, material transactions | Physical movement/transfer |
| **relational_act** | Social interactions, interpersonal conduct | Focus on relationship between parties |
| **trait_disposition** | Stable character traits, habitual patterns | Describes who someone IS, not what they DO |
| **mixed** | Multiple distinct behavior types in same span | 2+ clearly different forms present |

---

## Decision Rules for Common Confusions

### Rule 1: `mixed` vs `relational_act`
**Use `mixed` ONLY when the verse explicitly describes 2+ distinct behavior types.**

| Verse | Text Summary | Correct | Reasoning |
|-------|--------------|---------|-----------|
| 3:159 | "Be gentle, consult, then trust Allah" | `mixed` | Relational (gentle) + speech (consult) + inner (trust) |
| 4:36 | "Worship Allah, be good to parents..." | `mixed` | Worship + multiple relational acts |
| 60:8 | "Deal justly with non-combatants" | `relational_act` | Single relational instruction |

**Test**: Can you identify 2+ *different* behavior categories? If not → single form.

---

### Rule 2: `inner_state` vs `trait_disposition`
**Use `trait_disposition` for stable characteristics; `inner_state` for momentary states.**

| Verse | Text Summary | Correct | Reasoning |
|-------|--------------|---------|-----------|
| 2:5 | "They are upon guidance" | `trait_disposition` | Describes their established state |
| 4:145 | "Hypocrites in lowest depths" | `trait_disposition` | Describes their nature |
| 63:4 | "Their hearts are sealed" | `inner_state` | Describes current spiritual condition |

**Test**: Is this describing WHO they are (trait) or WHAT they're experiencing (state)?

---

### Rule 3: `physical_act` vs `relational_act`
**Focus on what the verse emphasizes - the ACTION or the RELATIONSHIP.**

| Verse | Text Summary | Correct | Reasoning |
|-------|--------------|---------|-----------|
| 2:191 | "Kill them where you find them" | `physical_act` | Emphasis on the physical action |
| 2:261 | "Spend in Allah's way" | `physical_act` | Emphasis on giving (transfer of wealth) |
| 9:34 | "Devour people's wealth wrongfully" | `physical_act` | Taking wealth is physical |
| 4:36 | "Be good to parents" | `relational_act` | Emphasis on relationship quality |

**Test**: Remove the other party - does the action still make sense?
- "Kill" still makes sense → physical_act
- "Be good to" doesn't make sense alone → relational_act

---

### Rule 4: `speech_act` vs `relational_act`
**Use `speech_act` when speaking IS the behavior; `relational_act` when speech serves a relationship.**

| Verse | Text Summary | Correct | Reasoning |
|-------|--------------|---------|-----------|
| 49:1 | "Do not raise voices above Prophet" | `speech_act` | About how to speak |
| 33:70 | "Speak straight words" | `speech_act` | About speech quality |
| 4:135 | "Stand firm for justice, testify" | `relational_act` | Testimony serves justice (relationship) |

---

### Rule 5: `inner_state` vs `speech_act`
**If the verse describes what someone SAYS about their inner state, code the PRIMARY behavior.**

| Verse | Text Summary | Correct | Reasoning |
|-------|--------------|---------|-----------|
| 63:6 | "Whether you ask forgiveness or not" | `speech_act` | About the act of asking |
| 3:8 | "Our Lord, do not let our hearts deviate" | `speech_act` | Dua is speech act |
| 2:10 | "In their hearts is disease" | `inner_state` | Describes their heart condition |

---

## Genuine Ambiguity Cases

Some verses are **legitimately ambiguous** - both annotators may be correct. In these cases:

1. **Prefer the more specific form** over `mixed`
2. **Document the ambiguity** in annotator notes
3. **Accept disagreement** as valid

Examples of genuine ambiguity:
- Verses describing inner states that manifest externally
- Commands that involve both speech and action
- Traits that are also current states

---

## Summary Decision Tree

```
1. Does the verse describe 2+ CLEARLY DISTINCT behavior types?
   → YES: mixed
   → NO: continue

2. Is the behavior primarily INTERNAL (beliefs, emotions, intentions)?
   → YES: Is it a stable trait or momentary state?
      → Stable: trait_disposition
      → Momentary: inner_state
   → NO: continue

3. Is SPEAKING the primary action?
   → YES: speech_act
   → NO: continue

4. Is PHYSICAL ACTION the primary behavior?
   → YES: physical_act
   → NO: continue

5. Is the RELATIONSHIP between parties the focus?
   → YES: relational_act
```

---

## Expected Impact

Applying these rules should:
- Reduce `mixed` overuse (Rule 1)
- Clarify trait vs state distinction (Rule 2)
- Resolve physical vs relational confusion (Rule 3)

**Target**: Improve κ from 0.703 to ~0.80+ without forcing agreement on genuinely ambiguous cases.
