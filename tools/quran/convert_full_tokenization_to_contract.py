"""
Convert a full Qur'an tokenization JSON into the repo's minimal contract format.

Why:
- We sometimes receive tokenized corpora in a "rich" structure (metadata + per-surah details).
- The project contract (see `data/quran/README.md` + `schemas/quran_tokenized_v1.schema.json`)
  expects a stable minimal structure:
    - quran_text_version
    - tokenization_id
    - surahs[] with {surah, ayat[]} where each ayah has {ayah, text, tokens[]}

This converter keeps extra metadata where possible while ensuring required fields exist.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional


class QuranTokenizationConvertError(ValueError):
    """Raised when a tokenization artifact cannot be converted to the contract format."""


def convert_source_to_contract(
    source: Dict[str, Any],
    quran_text_version: str = "uthmani_hafs_v1",
) -> Dict[str, Any]:
    """
    Convert a "full tokenization" JSON object to the repo contract format.

    Args:
        source (Dict[str, Any]): Input JSON object (already parsed).
        quran_text_version (str): Text version identifier for the output.

    Returns:
        Dict[str, Any]: Converted tokenization object that matches the contract schema.

    Raises:
        QuranTokenizationConvertError: If required fields are missing or inconsistent.
    """
    if not isinstance(source, dict):
        raise QuranTokenizationConvertError("Source must be a JSON object (dict).")

    metadata = source.get("metadata", {})
    tokenization_id = metadata.get("tokenization_id")
    if not tokenization_id or not isinstance(tokenization_id, str):
        raise QuranTokenizationConvertError("Missing or invalid source.metadata.tokenization_id.")

    surahs_in = source.get("surahs")
    if not isinstance(surahs_in, list) or not surahs_in:
        raise QuranTokenizationConvertError("Missing or invalid source.surahs (must be a non-empty list).")

    surahs_out: List[Dict[str, Any]] = []
    for s in surahs_in:
        if not isinstance(s, dict):
            raise QuranTokenizationConvertError("Invalid surah entry (expected object).")

        surah_number = s.get("surah_number")
        if not isinstance(surah_number, int) or not (1 <= surah_number <= 114):
            raise QuranTokenizationConvertError("Invalid surah_number in source.surahs[].")

        ayat_in = s.get("ayat")
        if not isinstance(ayat_in, list):
            raise QuranTokenizationConvertError(f"Invalid ayat list for surah {surah_number}.")

        ayat_out: List[Dict[str, Any]] = []
        for a in ayat_in:
            if not isinstance(a, dict):
                raise QuranTokenizationConvertError(f"Invalid ayah entry in surah {surah_number}.")

            ayah_number = a.get("ayah_number")
            text_ar = a.get("text_ar")
            tokens = a.get("tokens")

            if not isinstance(ayah_number, int) or ayah_number < 1:
                raise QuranTokenizationConvertError(f"Invalid ayah_number in surah {surah_number}.")
            if not isinstance(text_ar, str) or not text_ar:
                raise QuranTokenizationConvertError(f"Invalid text_ar in surah {surah_number}:{ayah_number}.")
            if not isinstance(tokens, list) or not tokens:
                raise QuranTokenizationConvertError(f"Invalid tokens in surah {surah_number}:{ayah_number}.")

            # Minimal per-token validation (cheap, fail-closed).
            for t in tokens:
                if not isinstance(t, dict):
                    raise QuranTokenizationConvertError(
                        f"Invalid token object in surah {surah_number}:{ayah_number}."
                    )
                if not isinstance(t.get("index"), int) or t["index"] < 0:
                    raise QuranTokenizationConvertError(
                        f"Invalid token.index in surah {surah_number}:{ayah_number}."
                    )
                if not isinstance(t.get("text"), str) or t["text"] == "":
                    raise QuranTokenizationConvertError(
                        f"Invalid token.text in surah {surah_number}:{ayah_number}."
                    )
                if not isinstance(t.get("start_char"), int) or not isinstance(t.get("end_char"), int):
                    raise QuranTokenizationConvertError(
                        f"Invalid token offsets in surah {surah_number}:{ayah_number}."
                    )

            ayah_out: Dict[str, Any] = {
                "ayah": ayah_number,
                "text": text_ar,
                "tokens": tokens,
            }

            # Keep useful extras if present.
            if "token_count" in a:
                ayah_out["token_count"] = a["token_count"]

            ayat_out.append(ayah_out)

        surah_out: Dict[str, Any] = {
            "surah": surah_number,
            "ayat": ayat_out,
        }
        # Keep names if present (schema allows extra fields).
        if "name_ar" in s:
            surah_out["name_ar"] = s["name_ar"]

        surahs_out.append(surah_out)

    out: Dict[str, Any] = {
        "quran_text_version": quran_text_version,
        "tokenization_id": tokenization_id,
        "surahs": surahs_out,
        "metadata": metadata,
    }
    return out


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json_minified(path: str, obj: Dict[str, Any]) -> None:
    """
    Write JSON to disk in a minified single-line form (to satisfy the <500 lines rule).

    Args:
        path (str): Output file path.
        obj (Dict[str, Any]): JSON-serializable object.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True, help="Input full tokenization JSON path.")
    parser.add_argument(
        "--out",
        dest="out_path",
        required=True,
        help="Output contract tokenization JSON path (will be written minified).",
    )
    parser.add_argument("--quran-text-version", default="uthmani_hafs_v1")
    args = parser.parse_args()

    source = _read_json(args.in_path)
    contract = convert_source_to_contract(source, quran_text_version=args.quran_text_version)
    write_json_minified(args.out_path, contract)
    print(f"Wrote contract tokenization -> {args.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


