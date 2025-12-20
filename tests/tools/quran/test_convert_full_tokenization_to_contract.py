from __future__ import annotations

import pytest

from tools.quran.convert_full_tokenization_to_contract import (
    QuranTokenizationConvertError,
    convert_source_to_contract,
)


def test_convert_source_to_contract_success_minimal() -> None:
    source = {
        "metadata": {"tokenization_id": "tok_v1"},
        "surahs": [
            {
                "surah_number": 1,
                "name_ar": "الفاتحة",
                "ayat": [
                    {
                        "ayah_number": 1,
                        "text_ar": "A B",
                        "token_count": 2,
                        "tokens": [
                            {"index": 0, "text": "A", "start_char": 0, "end_char": 1},
                            {"index": 1, "text": "B", "start_char": 2, "end_char": 3},
                        ],
                    }
                ],
            }
        ],
    }

    out = convert_source_to_contract(source, quran_text_version="uthmani_hafs_v1")
    assert out["quran_text_version"] == "uthmani_hafs_v1"
    assert out["tokenization_id"] == "tok_v1"
    assert out["surahs"][0]["surah"] == 1
    assert out["surahs"][0]["name_ar"] == "الفاتحة"
    assert out["surahs"][0]["ayat"][0]["ayah"] == 1
    assert out["surahs"][0]["ayat"][0]["text"] == "A B"
    assert out["surahs"][0]["ayat"][0]["token_count"] == 2
    assert out["surahs"][0]["ayat"][0]["tokens"][1]["start_char"] == 2


def test_convert_source_to_contract_fails_missing_tokenization_id() -> None:
    source = {"metadata": {}, "surahs": [{"surah_number": 1, "ayat": []}]}
    with pytest.raises(QuranTokenizationConvertError):
        convert_source_to_contract(source)


def test_convert_source_to_contract_fails_bad_token_offsets() -> None:
    source = {
        "metadata": {"tokenization_id": "tok_v1"},
        "surahs": [
            {
                "surah_number": 1,
                "ayat": [
                    {
                        "ayah_number": 1,
                        "text_ar": "A",
                        "tokens": [{"index": 0, "text": "A", "start_char": "0", "end_char": 1}],
                    }
                ],
            }
        ],
    }
    with pytest.raises(QuranTokenizationConvertError):
        convert_source_to_contract(source)


