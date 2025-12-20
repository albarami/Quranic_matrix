## Coding Manual v1 (outline)

This is an outline/skeleton. Expand into a full manual before pilot annotation.

---

## 1) Scope + principles

- Fail-closed: **no evidence → no label**
- Separate layers:
  - Qur’an-only textual signals vs. tafsir-supported disambiguation vs. optional juristic derivations
- Controlled IDs only (no free-text categories)

---

## 2) Span segmentation rules

### 2.1 Definitions

- A span is a **contiguous token range within one ayah**
- Token coordinates:
  - `span.token_start` is **inclusive**
  - `span.token_end` is **exclusive**

### 2.2 Segmentation decision rules (minimum)

- Prefer smallest meaning-bearing unit that supports the assertion.
- If a label depends on a contrast clause, include both clauses or use `linked_spans[]`.
- Store alternative boundaries in `span.alternative_boundaries[]` when debated.

---

## 3) Organ annotation playbook

### 3.1 When to tag `AX_ORGAN`

- Explicit organ mention: direct support with token anchor.
- Metaphor/metonymy: indirect support, must justify and record tafsir agreement.

### 3.2 Heart (`ORG_HEART`) semantic domains

Annotate `organ_semantic_domains[]` with one or more:
- physiological
- cognitive
- spiritual
- emotional

Tie-break: use `primary_organ_semantic_domain` if required by exports.

---

## 4) Normative textual layer: deontic decision tree

Record:
- `normative_textual.speech_mode`
- `normative_textual.evaluation`
- `normative_textual.quran_deontic_signal`

Minimum decision rules:
- Imperative form → `amr`
- Prohibition form → `nahy`
- Reward/praise cues → `targhib`
- Threat/blame cues → `tarhib`
- Otherwise → `khabar`

---

## 5) Negation patterns guide

### 5.1 “لا…ولكن / لكنهم …” encoding

- Create two assertions:
  1) Negated X (`negated=true`)
  2) Affirmed Y (`negated=false`)
- Anchor tokens to the negation and contrast clauses when possible.
- Use `intra_textual_references[]` with `relation_type="contrast"` when needed.

---

## 6) Periodicity rules

- Default `PER_UNKNOWN` unless explicit frequency lexemes/grammar exists.
- If frequency is known externally (Sunnah/ijma), store under `frequency_prescription` only.

---

## 7) Confidence + ESS

Store ESS per assertion:
- directness
- clarity
- specificity
- tafsir_agreement

Starter caps:
- Indirect + clarity < 0.7 → confidence ≤ 0.75
- Metaphor/metonymy → confidence ≤ 0.80 unless tafsir agreement is high
- Tafsir disputed → confidence ≤ 0.70 or set to unknown

---

## 8) Tafsir consultation protocol

Record:
- sources used
- agreement level (high/mixed/disputed)
- short note summarizing why it affects the label

---

## 9) Worked examples (required before pilot)

Target:
- 20+ full JSON examples
- include: expected use + edge case + failure case patterns

Store examples under `docs/coding_manual/examples/`.


