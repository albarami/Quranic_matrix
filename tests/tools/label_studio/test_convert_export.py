import pytest

from tools.label_studio.convert_export import ConversionError, convert_labelstudio_export


def _base_task() -> dict:
    return {
        "id": "QBM_000001",
        "data": {
            "id": "QBM_000001",
            "surah": 24,
            "ayah": 24,
            "raw_text_ar": "ABC DEF GHI",
            "quran_text_version": "uthmani_hafs_v1",
            "tokenization_id": "tok_v1",
            "tokens": [
                {"index": 0, "text": "ABC", "start_char": 0, "end_char": 3},
                {"index": 1, "text": "DEF", "start_char": 4, "end_char": 7},
                {"index": 2, "text": "GHI", "start_char": 8, "end_char": 11},
            ],
        },
        "annotations": [
            {
                "completed_by": "A1",
                "result": [
                    {"from_name": "span_selection", "value": {"start": 4, "end": 11, "labels": ["BEHAVIOR_SPAN"]}},
                    {"from_name": "boundary_confidence", "value": {"choices": ["certain"]}},
                    {"from_name": "behavior_form", "value": {"choices": ["speech_act"]}},
                    {"from_name": "agent_type", "value": {"choices": ["AGT_HUMAN_GENERAL"]}},
                    {"from_name": "agent_explicit", "value": {"choices": ["implicit"]}},
                    {"from_name": "speech_mode", "value": {"choices": ["informative"]}},
                    {"from_name": "evaluation", "value": {"choices": ["neutral"]}},
                    {"from_name": "quran_deontic_signal", "value": {"choices": ["khabar"]}},
                    {"from_name": "support_type", "value": {"choices": ["direct"]}},
                    {"from_name": "indication_tags", "value": {"choices": ["dalalah_mantuq"]}},
                    {"from_name": "justification_code", "value": {"choices": ["JST_EXPLICIT_MENTION"]}},
                    {"from_name": "polarity", "value": {"choices": ["neutral"]}},
                    {"from_name": "periodicity", "value": {"choices": ["PER_UNKNOWN"]}},
                    {"from_name": "grammatical_indicator", "value": {"choices": ["GRM_NONE"]}},
                    {"from_name": "tafsir_sources", "value": {"choices": ["IbnKathir"]}},
                    {"from_name": "tafsir_agreement", "value": {"choices": ["high"]}},
                    {"from_name": "confidence", "value": {"rating": 8}},
                    {"from_name": "justification_text", "value": {"text": "ok"}},
                ],
            }
        ],
    }


def test_convert_success_maps_normative_and_span_tokens() -> None:
    out = convert_labelstudio_export([_base_task()])
    assert len(out) == 1
    rec = out[0]
    assert rec["id"] == "QBM_000001"
    assert rec["span"]["token_start"] == 1
    assert rec["span"]["token_end"] == 3
    assert rec["normative_textual"]["speech_mode"] == "SPM_INFORMATIVE"
    assert rec["normative_textual"]["evaluation"] == "EVL_NEUTRAL"
    assert rec["normative_textual"]["quran_deontic_signal"] == "DNS_KHABAR"
    assert rec["periodicity"]["category"] == "PER_UNKNOWN"
    assert rec["periodicity"]["grammatical_indicator"] == "GRM_NONE"


def test_convert_fails_when_tokens_missing_offsets() -> None:
    task = _base_task()
    task["data"]["tokens"][0]["start_char"] = None
    with pytest.raises(ConversionError):
        convert_labelstudio_export([task])


def test_convert_fails_on_unknown_mapping_value() -> None:
    task = _base_task()
    # invalid speech_mode
    for item in task["annotations"][0]["result"]:
        if item["from_name"] == "speech_mode":
            item["value"]["choices"] = ["not_a_real_value"]
    with pytest.raises(ConversionError):
        convert_labelstudio_export([task])


