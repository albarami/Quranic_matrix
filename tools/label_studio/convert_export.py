"""
Label Studio export -> Quranic Behavior Matrix (QBM) span record conversion.

This module is intentionally stdlib-only to keep startup friction low.

Key constraint:
- Label Studio's text span selection typically yields character offsets (start/end).
  To convert to token indices, tasks MUST include token objects with start_char/end_char.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


class ConversionError(ValueError):
    """Raised when an export cannot be converted safely (fail-closed)."""


@dataclass(frozen=True)
class Token:
    index: int
    text: str
    start_char: int
    end_char: int


def _map_choice(mapping: Dict[str, str], value: Optional[str], field_name: str) -> Optional[str]:
    """
    Map a UI value to a controlled ID (or return None for missing).
    """

    if value is None:
        return None
    if value not in mapping:
        raise ConversionError(f"Unknown {field_name} value: {value!r}")
    return mapping[value]


def _first_choice(item: Dict[str, Any]) -> Optional[str]:
    choices = (item.get("value") or {}).get("choices")
    if not choices:
        return None
    return choices[0]


def _all_choices(item: Dict[str, Any]) -> List[str]:
    choices = (item.get("value") or {}).get("choices")
    if not choices:
        return []
    return list(choices)


def _taxonomy_values(item: Dict[str, Any]) -> List[str]:
    """
    Label Studio taxonomy output is typically nested lists; we flatten leaf IDs.
    """

    tax = (item.get("value") or {}).get("taxonomy") or []
    flat: List[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, str):
            flat.append(node)
            return
        if isinstance(node, list):
            for x in node:
                walk(x)
            return
        # unknown structure -> ignore (fail-closed handled elsewhere if needed)

    walk(tax)
    # keep only BEH_ leaves (ignore category labels like "Speech")
    return [x for x in flat if isinstance(x, str) and x.startswith("BEH_")]


def _parse_tokens(task_data: Dict[str, Any]) -> List[Token]:
    raw_tokens = task_data.get("tokens") or []
    tokens: List[Token] = []
    for t in raw_tokens:
        if t.get("start_char") is None or t.get("end_char") is None:
            raise ConversionError(
                "Task tokens must include start_char/end_char to convert span selections to token indices."
            )
        tokens.append(
            Token(
                index=int(t["index"]),
                text=str(t.get("text", "")),
                start_char=int(t["start_char"]),
                end_char=int(t["end_char"]),
            )
        )
    return tokens


def _char_span_to_token_span(tokens: List[Token], start_char: int, end_char: int) -> Tuple[int, int]:
    """
    Convert a character-offset span [start_char, end_char) into token indices [start_token, end_token).
    Includes any token that overlaps the character range.
    """

    if end_char < start_char:
        raise ConversionError("Span end_char must be >= start_char.")

    overlapping = [t for t in tokens if not (t.end_char <= start_char or t.start_char >= end_char)]
    if not overlapping:
        raise ConversionError("No tokens overlap the selected span; cannot map to token indices.")

    start_token = min(t.index for t in overlapping)
    end_token = max(t.index for t in overlapping) + 1
    return start_token, end_token


def _extract_text_span_char_offsets(result_items: Iterable[Dict[str, Any]]) -> Optional[Tuple[int, int]]:
    """
    Find the BEHAVIOR_SPAN selection and return (start_char, end_char) if present.
    """

    for item in result_items:
        if item.get("from_name") != "span_selection":
            continue
        value = item.get("value") or {}
        # Text selection results typically include start/end (char offsets) and labels list.
        start = value.get("start")
        end = value.get("end")
        if start is None or end is None:
            return None
        return int(start), int(end)
    return None


def convert_labelstudio_export(ls_export: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert Label Studio JSON export (list of tasks) to QBM span records.

    Fail-closed:
    - If required mappings or token offsets are missing, raises ConversionError.
    """

    # UI -> controlled ID mappings (frozen in vocab/*).
    speech_mode_map = {
        "command": "SPM_COMMAND",
        "prohibition": "SPM_PROHIBITION",
        "informative": "SPM_INFORMATIVE",
        "narrative": "SPM_NARRATIVE",
        "parable": "SPM_PARABLE",
        "unknown": "SPM_UNKNOWN",
    }
    evaluation_map = {
        "praise": "EVL_PRAISE",
        "blame": "EVL_BLAME",
        "warning": "EVL_WARNING",
        "promise": "EVL_PROMISE",
        "neutral": "EVL_NEUTRAL",
        "mixed": "EVL_MIXED",
        "unknown": "EVL_UNKNOWN",
    }
    deontic_map = {
        "amr": "DNS_AMR",
        "nahy": "DNS_NAHY",
        "targhib": "DNS_TARGHIB",
        "tarhib": "DNS_TARHIB",
        "khabar": "DNS_KHABAR",
    }
    support_type_map = {"direct": "SUP_DIRECT", "indirect": "SUP_INDIRECT"}
    indication_map = {
        "dalalah_mantuq": "IND_DALALAH_MANTUQ",
        "dalalah_mafhum": "IND_DALALAH_MAFHUM",
        "narrative_inference": "IND_NARRATIVE_INFERENCE",
        "metaphor_metonymy": "IND_METAPHOR_METONYMY",
        "sabab_nuzul_used": "IND_SABAB_NUZUL_USED",
    }
    negation_type_map = {
        "absolute": "NEG_ABSOLUTE",
        "conditional": "NEG_CONDITIONAL",
        "exceptionless_affirmation": "NEG_EXCEPTIONLESS_AFFIRMATION",
        "unknown": "NEG_UNKNOWN",
    }
    polarity_map = {
        "positive": "POL_POSITIVE",
        "negative": "POL_NEGATIVE",
        "neutral": "POL_NEUTRAL",
        "mixed": "POL_MIXED",
        "unknown": "POL_UNKNOWN",
    }
    tafsir_agreement_map = {"high": "TAF_HIGH", "mixed": "TAF_MIXED", "disputed": "TAF_DISPUTED"}

    converted: List[Dict[str, Any]] = []

    for task in ls_export:
        data = task.get("data") or {}
        annotations = task.get("annotations") or []
        if not annotations:
            continue

        ann = annotations[0]
        result_items = ann.get("result") or []

        tokens = _parse_tokens(data)
        char_span = _extract_text_span_char_offsets(result_items)
        if char_span is None:
            raise ConversionError("Missing span_selection start/end in Label Studio results.")
        span_token_start, span_token_end = _char_span_to_token_span(tokens, char_span[0], char_span[1])

        # Collect scalar fields from results
        behavior_form: str = "unknown"
        agent_type: str = "AGT_UNKNOWN"
        agent_explicit: Optional[bool] = None
        thematic_constructs: List[str] = []
        behavior_concepts: List[str] = []

        normative_speech_mode: Optional[str] = None
        normative_eval: Optional[str] = None
        normative_deontic: Optional[str] = None

        support_type: Optional[str] = None
        indication_tags: List[str] = []
        justification_code: Optional[str] = None
        negated: Optional[bool] = None
        negation_type: Optional[str] = None
        polarity: Optional[str] = None
        confidence: Optional[float] = None
        justification_text: Optional[str] = None

        periodicity: Optional[str] = None
        grammatical_indicator: Optional[str] = None

        tafsir_sources: List[str] = []
        tafsir_agreement: Optional[str] = None

        organs: List[str] = []
        heart_domains: List[str] = []
        organ_role: Optional[str] = None

        systemic: List[str] = []
        primary_systemic: Optional[str] = None
        situational: Optional[str] = None
        situational_domains: List[str] = []
        spatial: Optional[str] = None
        temporal: Optional[str] = None

        action_class: Optional[str] = None
        action_textual_eval: Optional[str] = None

        notes: Optional[str] = None

        for item in result_items:
            from_name = item.get("from_name")
            if from_name == "behavior_form":
                behavior_form = _first_choice(item) or "unknown"
            elif from_name == "agent_type":
                agent_type = _first_choice(item) or "AGT_UNKNOWN"
            elif from_name == "agent_explicit":
                v = _first_choice(item)
                if v == "explicit":
                    agent_explicit = True
                elif v == "implicit":
                    agent_explicit = False
            elif from_name == "thematic_constructs":
                thematic_constructs = _all_choices(item)
            elif from_name == "behavior_concepts":
                behavior_concepts = _taxonomy_values(item)
            elif from_name == "speech_mode":
                normative_speech_mode = _map_choice(speech_mode_map, _first_choice(item), "speech_mode")
            elif from_name == "evaluation":
                normative_eval = _map_choice(evaluation_map, _first_choice(item), "evaluation")
            elif from_name == "quran_deontic_signal":
                normative_deontic = _map_choice(deontic_map, _first_choice(item), "quran_deontic_signal")
            elif from_name == "support_type":
                support_type = _map_choice(support_type_map, _first_choice(item), "support_type")
            elif from_name == "indication_tags":
                indication_tags = [_map_choice(indication_map, x, "indication_tags") for x in _all_choices(item)]
            elif from_name == "justification_code":
                justification_code = _first_choice(item)
            elif from_name == "negated":
                v = _first_choice(item)
                if v == "true":
                    negated = True
                elif v == "false":
                    negated = False
            elif from_name == "negation_type":
                negation_type = _map_choice(negation_type_map, _first_choice(item), "negation_type")
            elif from_name == "polarity":
                polarity = _map_choice(polarity_map, _first_choice(item), "polarity")
            elif from_name == "confidence":
                # Label Studio Rating uses 1..maxRating, stored in value["rating"]
                rating = (item.get("value") or {}).get("rating")
                if rating is not None:
                    confidence = float(rating) / 10.0
            elif from_name == "justification_text":
                txt = (item.get("value") or {}).get("text")
                if isinstance(txt, str):
                    justification_text = txt
            elif from_name == "notes":
                txt = (item.get("value") or {}).get("text")
                if isinstance(txt, str):
                    notes = txt
            elif from_name == "periodicity":
                periodicity = _first_choice(item)
            elif from_name == "grammatical_indicator":
                grammatical_indicator = _first_choice(item)
            elif from_name == "tafsir_sources":
                tafsir_sources = _all_choices(item)
            elif from_name == "tafsir_agreement":
                tafsir_agreement = _map_choice(tafsir_agreement_map, _first_choice(item), "tafsir_agreement")
            elif from_name == "organs":
                organs = _all_choices(item)
            elif from_name == "heart_domains":
                heart_domains = _all_choices(item)
            elif from_name == "organ_role":
                organ_role = _first_choice(item)
            elif from_name == "systemic":
                systemic = _all_choices(item)
            elif from_name == "primary_systemic":
                primary_systemic = _first_choice(item)
            elif from_name == "situational":
                situational = _first_choice(item)
            elif from_name == "situational_domains":
                situational_domains = _all_choices(item)
            elif from_name == "spatial":
                spatial = _first_choice(item)
            elif from_name == "temporal":
                temporal = _first_choice(item)
            elif from_name == "action_class":
                action_class = _first_choice(item)
            elif from_name == "action_textual_eval":
                action_textual_eval = _first_choice(item)

        if normative_speech_mode is None or normative_eval is None or normative_deontic is None:
            raise ConversionError("Missing normative_textual fields (speech_mode/evaluation/quran_deontic_signal).")
        if support_type is None:
            raise ConversionError("Missing support_type.")

        record_id = data.get("id") or task.get("id")
        if record_id is None:
            raise ConversionError("Missing task id.")

        # Build assertions (minimal, axis-scoped); evidence anchors use token indices when possible.
        assertions: List[Dict[str, Any]] = []
        a_i = 1

        def add_assertion(axis: str, value: str, extra: Optional[Dict[str, Any]] = None) -> None:
            nonlocal a_i
            payload: Dict[str, Any] = {
                "assertion_id": f"{record_id}_A{a_i:03d}",
                "axis": axis,
                "value": value,
                "support_type": "direct" if support_type == "SUP_DIRECT" else "indirect",
                "indication_tags": indication_tags,
                "evidence_anchor": {"token_start": span_token_start, "token_end": span_token_end},
                "justification_code": justification_code,
                "justification": justification_text or "",
                "confidence": confidence,
                "polarity": polarity or "POL_UNKNOWN",
                "negated": bool(negated) if negated is not None else False,
                "negation_type": negation_type or "NEG_UNKNOWN",
            }
            if extra:
                payload.update(extra)
            assertions.append(payload)
            a_i += 1

        # Organic assertions
        for org in organs:
            extra = {"organ_role": organ_role or "ROLE_UNKNOWN"}
            if org == "ORG_HEART":
                extra["organ_semantic_domains"] = heart_domains
                extra["primary_organ_semantic_domain"] = heart_domains[0] if heart_domains else "DOM_COGNITIVE"
            add_assertion("AX_ORGAN", org, extra=extra)

        # Situational/systemic/spatial/temporal assertions are axis-value assertions
        if situational:
            add_assertion("AX_SITUATIONAL", situational, extra={"situational_domains": situational_domains})
        for sysv in systemic:
            add_assertion("AX_SYSTEMIC", sysv, extra={"primary_systemic": primary_systemic})
        if spatial:
            add_assertion("AX_SPATIAL", spatial)
        if temporal:
            add_assertion("AX_TEMPORAL", temporal)
        if action_class:
            add_assertion("AX_ACTION_CLASS", action_class)
        if action_textual_eval:
            add_assertion("AX_ACTION_TEXTUAL_EVAL", action_textual_eval)

        out = {
            "id": str(record_id),
            "quran_text_version": data.get("quran_text_version", "uthmani_hafs_v1"),
            "tokenization_id": data.get("tokenization_id", "tok_v1"),
            "reference": {"surah": int(data.get("surah")), "ayah": int(data.get("ayah"))},
            "span": {
                "token_start": span_token_start,
                "token_end": span_token_end,
                "raw_text_ar": data.get("raw_text_ar"),
                "boundary_confidence": _first_choice(
                    next((x for x in result_items if x.get("from_name") == "boundary_confidence"), {})
                )
                or "certain",
                "alternative_boundaries": [],
            },
            "behavior": {"concepts": behavior_concepts, "form": behavior_form},
            "thematic_constructs": thematic_constructs,
            "meta_discourse_constructs": [],
            "agent": {
                "type": agent_type,
                "explicit": bool(agent_explicit) if agent_explicit is not None else False,
                "support_type": "direct" if support_type == "SUP_DIRECT" else "indirect",
                "evidence_anchor": {"token_start": span_token_start, "token_end": span_token_end},
                "note": "",
            },
            "tafsir": {
                "sources_used": tafsir_sources,
                "agreement_level": tafsir_agreement or "TAF_HIGH",
                "consultation_trigger": "baseline",
                "notes": "",
            },
            "assertions": assertions,
            "periodicity": {
                "category": periodicity or "PER_UNKNOWN",
                "grammatical_indicator": grammatical_indicator or "GRM_NONE",
                "support_type": "direct" if support_type == "SUP_DIRECT" else "indirect",
                "indication_tags": indication_tags,
                "evidence_anchor": {"token_start": span_token_start, "token_end": span_token_end},
                "justification_code": justification_code,
                "justification": justification_text or "",
                "confidence": confidence,
                "ess": None,
            },
            "normative_textual": {
                "speech_mode": normative_speech_mode,
                "evaluation": normative_eval,
                "quran_deontic_signal": normative_deontic,
                "support_type": "direct" if support_type == "SUP_DIRECT" else "indirect",
                "note": "",
            },
            "review": {"status": "REV_DRAFT", "annotator_id": ann.get("completed_by"), "reviewer_id": "", "notes": notes or ""},
        }

        converted.append(out)

    return converted


