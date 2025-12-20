"""
Build pilot_50 selections JSONL from a local Tanzil Uthmani XML file.

Why:
- Keep the repo lightweight and avoid committing full corpora.
- Reproducible generation from a trusted source file + references list.

Output:
- JSONL where each line is a selection record containing:
  - reference, surah_number, surah_name, ayah_number, text_ar, token_count, tokens[]

Tokenization:
- whitespace_split with sequential char offsets.
"""

from __future__ import annotations

import argparse
import json
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


class PilotBuildError(ValueError):
    """Fail-closed build error for pilot dataset generation."""


@dataclass(frozen=True)
class AyahRef:
    surah: int
    ayah: int


def parse_refs(lines: Iterable[str]) -> List[AyahRef]:
    refs: List[AyahRef] = []
    for raw in lines:
        s = raw.strip()
        if not s:
            continue
        if ":" not in s:
            raise PilotBuildError(f"Invalid reference (expected surah:ayah): {s!r}")
        a, b = s.split(":", 1)
        try:
            surah = int(a)
            ayah = int(b)
        except ValueError as e:
            raise PilotBuildError(f"Invalid numeric reference: {s!r}") from e
        refs.append(AyahRef(surah=surah, ayah=ayah))
    return refs


def whitespace_tokenize_with_offsets(text: str) -> Tuple[int, List[Dict]]:
    """
    Whitespace-split tokenization with char offsets.

    Offsets are based on the raw string:
    - start_char inclusive
    - end_char exclusive
    """

    tokens: List[Dict] = []
    i = 0
    idx = 0
    n = len(text)

    while i < n:
        # skip spaces
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        start = i
        while i < n and not text[i].isspace():
            i += 1
        end = i
        tok = text[start:end]
        tokens.append({"index": idx, "text": tok, "start_char": start, "end_char": end})
        idx += 1

    return idx, tokens


def load_tanzil_xml_index(xml_path: str) -> Dict[Tuple[int, int], Dict]:
    """
    Index ayah text from Tanzil XML:
    <quran><sura index=".." name=".."><aya index=".." text=".." /></sura></quran>
    """

    tree = ET.parse(xml_path)
    root = tree.getroot()
    if root.tag != "quran":
        raise PilotBuildError(f"Unexpected root tag: {root.tag!r} (expected 'quran').")

    index: Dict[Tuple[int, int], Dict] = {}
    for sura_el in root.findall("sura"):
        surah_num = int(sura_el.attrib["index"])
        surah_name = sura_el.attrib.get("name", "")
        for aya_el in sura_el.findall("aya"):
            ayah_num = int(aya_el.attrib["index"])
            text = aya_el.attrib.get("text")
            if text is None:
                continue
            index[(surah_num, ayah_num)] = {"surah_name": surah_name, "text_ar": text}

    return index


def build_selections(xml_index: Dict[Tuple[int, int], Dict], refs: List[AyahRef]) -> List[Dict]:
    selections: List[Dict] = []
    for r in refs:
        key = (r.surah, r.ayah)
        if key not in xml_index:
            raise PilotBuildError(f"Reference not found in XML: {r.surah}:{r.ayah}")
        meta = xml_index[key]
        text_ar = meta["text_ar"]
        token_count, tokens = whitespace_tokenize_with_offsets(text_ar)
        selections.append(
            {
                "reference": f"{r.surah}:{r.ayah}",
                "surah_number": r.surah,
                "surah_name": meta.get("surah_name", ""),
                "ayah_number": r.ayah,
                "text_ar": text_ar,
                "token_count": token_count,
                "tokens": tokens,
            }
        )
    return selections


def write_jsonl(out_path: str, selections: List[Dict]) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for obj in selections:
            f.write(json.dumps(obj, ensure_ascii=False))
            f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", required=True, help="Path to Tanzil Uthmani XML (quran-uthmani.xml).")
    parser.add_argument("--refs", required=True, help="Path to references list (one 'surah:ayah' per line).")
    parser.add_argument("--out", required=True, help="Output JSONL path.")
    args = parser.parse_args()

    with open(args.refs, "r", encoding="utf-8") as f:
        refs = parse_refs(f.readlines())
    xml_index = load_tanzil_xml_index(args.xml)
    selections = build_selections(xml_index, refs)
    write_jsonl(args.out, selections)
    print(f"Wrote {len(selections)} selections -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


