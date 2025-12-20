## Quranic Human‑Behavior Classification Matrix (Usul al‑Fiqh–Aligned)
### Standalone methodology and implementation specification (enhanced using Bouzidani’s new paper version)

---

## 0) Orientation (why this specification exists)
This specification defines a **Qur’an‑grounded, audit‑grade classification matrix for human behavior** that is **Usul‑aligned** and **fail‑closed**:

- **No evidence → no label** (and no “derived meaning” masquerading as text).
- **Direct vs indirect** is tracked per label.
- **Qur’an‑only signals** are separated from **juristic derivations**.
- **Controlled IDs** prevent drift across annotators and institutions.

This version incorporates key conceptual reinforcement from Ibrahim Bouzidani’s updated study on building a Qur’anic behavior matrix in its Islamic psychology context:
- **فطرة البصيرة** (fitrah‑based moral insight) + **موضوعية الأخلاق** (objective ethics) as a grounding for why a Qur’an‑based behavioral taxonomy is not merely descriptive.
- A **three‑world model** of human experience (sensory / psychic‑cognitive / spiritual) as an explanatory scaffold for “internal vs external” behavior and the role of intention.
- A tighter **عمل/نية/غريزة** framing: behavior as “action” (عمل) can be morally ranked (صالح/غير صالح), and **intention can transform routine/neutral acts into worship‑value** (without collapsing text into fiqh).

It also adopts the paper’s warning about **importing psychological taxonomies without their philosophical premises**. Therefore this specification:
- makes philosophical assumptions explicit (Qur’anic anthropology vs materialist reduction),
- keeps “crosswalks” to modern taxonomies optional and clearly marked as analytic overlays,
- prevents drift by requiring evidence anchors and controlled IDs.

---

## 1) Purpose and scope

### 1.1 Purpose
Produce a structured dataset of **Qur’anic spans** annotated across:
- five major contexts/axes (organic, situational, systemic, spatial, temporal),
- overlays (intention, periodicity, behavioral classification),
- normative textual signals (Qur’an‑only),
- optional juristic derivations (separate, fully sourced).

### 1.2 Intended uses
- **Scholarly**: traceable behavioral and thematic mapping of Qur’anic discourse.
- **Annotation infrastructure**: multi‑institution consistency with measurable IAA.
- **Computational research**: graph + retrieval + analytics without over‑interpretation.

### 1.3 Non‑goals
- Not a fatwa engine.
- Not “Qur’an‑only fiqh.”
- Not replacing tafsir; tafsir is metadata for stabilization and audit.

---

## 2) Epistemic foundations (Usul + Islamic psychology alignment)

### 2.1 Usul boundary: what we claim vs what we store
- **Textual layer**: what the Qur’an indicates in wording/structure.
- **Interpretive layer**: tafsir‑supported disambiguation and controlled indirect inference.
- **Juristic layer**: legal rulings and usul reasoning are optional and must be isolated.

Paper scope alignment note (important):
- The updated Bouzidani paper builds its matrix using **Qur’an + Sunnah + inherited Islamic sciences** as an integrated reference frame.
- This dataset remains **Qur’an‑grounded** and records non‑Qur’anic clarifications (e.g., ritual frequency prescriptions) only in explicitly separated fields (`frequency_prescription` and any optional “external support” blocks). This preserves auditability and prevents Qur’an‑only over‑claims.

### 2.2 Fitrah‑based moral insight and objective ethics
The dataset assumes:
- Humans have **capacity for being affected and affecting** (قابلية التأثر/التأثير),
- Behavior can be evaluated against an **objective moral frame** (موضوعية الأخلاق),
- The Qur’an provides a coherent, comprehensive discourse that can ground a matrix of contexts.

This is not a claim that every label is a ruling; it is a claim that the Qur’an contains **behavioral discourse** that can be structured with evidence discipline.

### 2.3 Three‑world model (sensory / psychic / spiritual)
For operational clarity in annotation:
- **Sensory world**: observable, embodied actions and perceptions.
- **Psychic/cognitive world**: thoughts, emotions, motivations, dispositions.
- **Spiritual world**: worship‑orientation, accountability, and intention‑linked valuation.

This model supports:
- why “internal vs external” is not merely visibility,
- why intention is a separate overlay,
- why “organ” mentions can carry cognitive/spiritual semantics (esp. قلب).

Paper‑anchored note:
- The updated paper cites (as conceptual support) a three‑world framing associated with Ibn Khaldun: sensory perception, cognitive/intellectual apprehension beyond sensory, and a higher spiritual/ghaybi domain (e.g., intentions, orientations, worship‑value). This document uses that framing as **an annotation scaffold**, not as a claim that every span encodes all three.

### 2.4 “Action (عمل)” framing and intention transformation (نية)
Bouzidani emphasizes that Qur’anic discourse often uses **عمل** (actions) rather than the modern term “behavior.” Operationally:
- Store **behavior concepts** as labels.
- Store **textual evaluation** separately.
- Permit an optional “action valuation” axis only when explicitly supported.

Crucially: **intention may elevate a routine act into worship‑value**, but this transformation is not “Qur’an‑only”; it may require Sunnah/fiqh framing. Therefore:
- The dataset stores intention as an overlay.
- Any frequency/ritual prescription derived from Sunnah is stored separately.

### 2.5 Method: inductive qualitative content analysis (paper-aligned)
The updated paper explicitly frames the approach as **Inductive Content Analysis** (often discussed as qualitative content analysis), operationalized here as:
- **Comprehension reading**: understand the texts under study in context.
- **Identify the “big picture”**: define the major contexts/axes that recur.
- **Extract precise meaning-bearing texts**: select spans that directly express the phenomenon.
- **Focus on qualitative points**: encode the recurring qualitative dimensions as controlled labels.
- **Re-compose a knowledge product**: synthesize the extracted categories into a stable matrix + coding manual.

This maps to our workflow as: span segmentation → controlled vocabularies → double-annotation → IAA → manual revision → scale.

---

## 3) Conceptual model

### 3.1 Unit of analysis: span
Primary unit is a **verse‑span**: a contiguous, token‑bounded segment within a single ayah.
- Multi‑ayah discourse is represented via linking (see `linked_spans[]`).

### 3.2 Three label families (do not conflate)
- **Behavior concepts** (`BEH_*`): actions, traits, dispositions, inner states.
- **Thematic constructs** (`THM_*`): frames/themes (accountability, testimony, reward/punishment).
- **Meta‑Qur’anic discourse constructs** (`META_*`, optional): revelation process, recitation mode, rhetoric.

### 3.3 Behavior form taxonomy (controlled)
`behavior.form` captures the structural form of the behavior label:
- `speech_act`
- `physical_act`
- `inner_state`
- `trait_disposition`
- `relational_act`
- `omission`
- `mixed`
- `unknown`

### 3.4 Quranic behavior classification overlay (from paper framing)
To align with the updated paper’s “غريزة/عمل صالح/عمل غير صالح + نية” framing without collapsing into juristic rulings, use an overlay axis:

- **Action class** (`AX_ACTION_CLASS`):
  - `ACT_INSTINCTIVE_OR_AUTOMATIC` (غريزي/لا إرادي)
  - `ACT_VOLITIONAL` (إرادي/مكتسب)
  - `ACT_UNKNOWN`

- **Action moral evaluation (textual)** (`AX_ACTION_TEXTUAL_EVAL`):
  - `EVAL_SALIH` | `EVAL_SAYYI` | `EVAL_NEUTRAL` | `EVAL_NOT_APPLICABLE` | `EVAL_UNKNOWN`

Rules:
- Only assign `EVAL_SALIH/EVAL_SAYYI` when the span provides **explicit evaluative cues** (praise/blame, promise/threat, etc.) or when tafsir agreement is high and the evaluation is clearly part of the meaning.
- Do not turn this into juristic status (wajib/haram) unless using the juristic layer.
- If `ACT_INSTINCTIVE_OR_AUTOMATIC`, default `AX_ACTION_TEXTUAL_EVAL` to `EVAL_NOT_APPLICABLE` unless the text explicitly evaluates the act (rare).

### 3.5 Behavior dynamics (stimulus → internal arousal → response) (optional, paper-aligned)
The updated paper emphasizes a behavioral sequence: **stimulus → internal arousal/activation → response**, and identifies three practical pillars for behavior analysis: **stimulus (المثير)**, **goal/purpose (الغاية/الهدف)**, and **intention (النية)**.

To support this without forcing labels where the Qur’an does not specify them, this spec introduces optional assertion axes (often left `unknown`):
- `AX_STIMULUS_MODALITY`: `STM_SENSORY` | `STM_PSYCHIC` | `STM_SPIRITUAL` | `STM_UNKNOWN`
- `AX_GOAL_ORIENTATION`: `GOL_WORLDLY` | `GOL_OTHERWORLDLY` | `GOL_MIXED` | `GOL_UNKNOWN`
- `AX_INTENTION_PRESENCE`: `INT_PRESENT` | `INT_ABSENT` | `INT_UNKNOWN`

These axes are **not required** for the pilot unless your annotation team can justify them with clear anchors and agreement.

**Annotator guidance (to resolve “intention tension”):**
- `AX_INTENTION_PRESENCE` is ONLY for **textually foregrounded intention** (explicit “يريدون/أراد/قصد/ابتغاء/وجه الله/يراءون…”-type signals).
- Do **not** infer “the agent must have intended the act” from the mere existence of an act. Default to `INT_UNKNOWN`.
- The **quality** of intention (sincerity vs showing off vs worldly) is captured in the intention overlay (not by guessing `INT_PRESENT`).

> ⚠️ **WARNING FOR IMPLEMENTERS (research-grade axes):**
>
> These three axes (`AX_STIMULUS_MODALITY`, `AX_GOAL_ORIENTATION`, `AX_INTENTION_PRESENCE`) are **HARD** and will remain `unknown` in the vast majority of spans.
>
> **Recommended for v1 pilot:**
> - Do NOT include them in double-annotation unless:
>   - you have ≥20 calibrated examples per axis,
>   - and you measured IAA ≥ 0.60 for that axis in a micro‑pilot.
>
> **Default stance:** keep them as optional fields but exclude them from v1 “Gold” export profiles unless explicitly enabled.

---

## 4) The matrix (axes)

### 4.1 Organic axis (عضوي/بيولوجي)
Captures organ references when framed as:
- tool of action,
- tool of perception,
- witness/accountability,
- metaphor for cognition/faith/emotion.

Special rule (paper‑motivated): when `ORG_HEART`, annotate `organ_semantic_domains[]`.

Paper‑anchored descriptive statistics (useful for coverage audits, not as a label rule):
- The updated paper reports **62 human organs** referenced across **645 ayat** (as an aggregate count). This can be used as a *sanity target* when building organ vocabularies and coverage audits.

**Coverage audit target (paper‑sourced, non‑normative):**
- Use the 62/645 statistic as a sanity check, not a quota.
- Example heuristic: if a pilot samples ~200 ayat/spans and yields **very low** organ mentions (e.g., <15), investigate selection bias or extraction gaps.

### 4.2 Situational axis (موضعي/حالي)
Captures internal/external:
- **external (ظاهر)**: speech/bodily acts,
- **internal (باطن)**: belief, intention, emotion, cognition.

Optional domain tags (when supported): emotional / spiritual / cognitive / psychological / social.

### 4.3 Systemic axis (نسقي)
Multi‑label frame:
- `SYS_SELF` (النفس)
- `SYS_CREATION` (الخلق)
- `SYS_GOD` (الخالق)
- `SYS_COSMOS` (الكون)
- `SYS_LIFE` (الحياة)

Optional `primary_systemic` with tie‑break rules in the coding manual.

### 4.4 Spatial axis (مكاني/جغرافي)
- `LOC_HOME` | `LOC_WORK` | `LOC_PUBLIC` | `LOC_UNKNOWN`

### 4.5 Temporal axis (زماني)
- `TMP_MORNING` | `TMP_NOON` | `TMP_AFTERNOON` | `TMP_NIGHT` | `TMP_UNKNOWN`

---

## 5) Evidence policy (Usul‑aligned)

### 5.1 Evidence is per assertion, not per span
All labels are stored as **assertions** with their own evidence metadata.

### 5.1.1 Behavior concepts and construct lists are derived caches (schema integrity rule)
For schema integrity and auditability:
- `behavior.concepts[]`, `thematic_constructs[]`, and `meta_discourse_constructs[]` are OPTIONAL **derived caches** for convenience.
- If present, each item MUST have a corresponding assertion in `assertions[]` using:
  - `AX_BEHAVIOR_CONCEPT` → `BEH_*`
  - `AX_THEMATIC_CONSTRUCT` → `THM_*`
  - `AX_META_DISCOURSE_CONSTRUCT` → `META_*`
- Exports MUST be fail‑closed if there is a mismatch between caches and assertions.

### 5.2 Controlled IDs (anti‑drift)
All categorical values MUST be stored as controlled IDs:
- axes (`AX_*`)
- systemic (`SYS_*`)
- spatial (`LOC_*`)
- temporal (`TMP_*`)
- agents (`AGT_*`)
- intentions (`INT_*`)
- periodicity (`PER_*`)
- behavior/construct/meta (`BEH_*`, `THM_*`, `META_*`)

### 5.3 Support type
- `direct`: explicit wording/structure.
- `indirect`: implication, narrative inference, metaphor/metonymy, context reasoning.

### 5.4 Indication tags (starter set)
- `dalalah_mantuq`
- `dalalah_mafhum`
- `narrative_inference`
- `metaphor_metonymy`
- `sabab_nuzul_used`

### 5.5 Polarity and negation
Each assertion includes:
- `negated`: true|false
- `negation_type`: absolute | conditional | exceptionless_affirmation | unknown
- `polarity`: positive|negative|neutral|mixed|unknown

### 5.6 Evidence anchors: token coordinate semantics
- `span.token_start` **inclusive**
- `span.token_end` **exclusive**
- `evidence_anchor` uses the **same ayah‑level token index space**.

`evidence_anchor` SHOULD be a token range:

```json
{"evidence_anchor": {"token_start": 12, "token_end": 15}}
```

If truly no anchor exists (purely contextual inference), `evidence_anchor` may be `null` and must be justified.

### 5.7 Justification (required for indirect)
- `justification_code` (`JST_*`)
- `justification` (short text)


