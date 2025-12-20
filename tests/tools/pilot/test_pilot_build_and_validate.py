import xml.etree.ElementTree as ET

import pytest

from tools.pilot.build_pilot_50_from_xml import (
    PilotBuildError,
    build_selections,
    load_tanzil_xml_index,
    parse_refs,
    whitespace_tokenize_with_offsets,
)
from tools.pilot.build_label_studio_tasks_from_pilot import build_tasks
from tools.pilot.validate_pilot_selections import PilotValidationError, validate_selection


def test_whitespace_tokenize_with_offsets_roundtrip() -> None:
    text = "A  B C"
    count, tokens = whitespace_tokenize_with_offsets(text)
    assert count == 3
    assert tokens[0]["text"] == "A"
    assert tokens[1]["text"] == "B"
    assert tokens[2]["text"] == "C"
    for t in tokens:
        assert text[t["start_char"] : t["end_char"]] == t["text"]


def test_parse_refs_ok_and_fail() -> None:
    refs = parse_refs(["2:3", " 4:1 "])
    assert [(r.surah, r.ayah) for r in refs] == [(2, 3), (4, 1)]
    with pytest.raises(PilotBuildError):
        parse_refs(["2-3"])


def test_build_selections_from_minimal_xml_and_validate() -> None:
    xml = """<?xml version="1.0" encoding="utf-8"?>
    <quran>
      <sura index="2" name="البقرة">
        <aya index="3" text="X Y" />
      </sura>
    </quran>"""
    # Write to a temp file because load_tanzil_xml_index reads a path
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "q.xml")
        with open(p, "w", encoding="utf-8") as f:
            f.write(xml)
        idx = load_tanzil_xml_index(p)
        refs = parse_refs(["2:3"])
        selections = build_selections(idx, refs)
        assert selections[0]["reference"] == "2:3"
        validate_selection(selections[0])


def test_validate_selection_fails_on_bad_offsets() -> None:
    sel = {
        "reference": "1:1",
        "surah_number": 1,
        "ayah_number": 1,
        "text_ar": "ABC",
        "token_count": 1,
        "tokens": [{"index": 0, "text": "ABC", "start_char": 0, "end_char": 2}],
    }
    with pytest.raises(PilotValidationError):
        validate_selection(sel)


def test_build_label_studio_tasks_wraps_data_and_qbm_id() -> None:
    selections = [
        {
            "reference": "2:3",
            "surah_number": 2,
            "surah_name": "البقرة",
            "ayah_number": 3,
            "text_ar": "A B",
            "token_count": 2,
            "tokens": [
                {"index": 0, "text": "A", "start_char": 0, "end_char": 1},
                {"index": 1, "text": "B", "start_char": 2, "end_char": 3},
            ],
        }
    ]
    tasks = build_tasks(selections, qbm_id_prefix="QBM", qbm_id_width=5, outer_id_start=1)
    assert tasks[0]["id"] == 1
    assert tasks[0]["data"]["id"] == "QBM_00001"
    assert tasks[0]["data"]["surah"] == 2
    assert tasks[0]["data"]["surah_name"] == "البقرة"
    assert tasks[0]["data"]["reference"] == "2:3"
    assert tasks[0]["data"]["token_count"] == 2


