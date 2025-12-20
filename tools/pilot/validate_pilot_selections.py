"""
Validation utilities for pilot selection JSONL.

Fail-closed: raises ValueError on any inconsistency.
"""

from __future__ import annotations

import json
from typing import Dict, Iterable, List, Tuple


class PilotValidationError(ValueError):
    """Raised when pilot selections fail validation."""


def validate_selection(selection: Dict) -> None:
    required = ["reference", "surah_number", "ayah_number", "text_ar", "token_count", "tokens"]
    for k in required:
        if k not in selection:
            raise PilotValidationError(f"Missing field: {k}")

    text = selection["text_ar"]
    tokens = selection["tokens"]
    if not isinstance(tokens, list):
        raise PilotValidationError("tokens must be a list")
    if selection["token_count"] != len(tokens):
        raise PilotValidationError("token_count does not match len(tokens)")

    for i, tok in enumerate(tokens):
        if tok.get("index") != i:
            raise PilotValidationError("token indices must be contiguous starting at 0")
        for k in ["text", "start_char", "end_char"]:
            if k not in tok:
                raise PilotValidationError(f"token missing {k}")
        s = tok["start_char"]
        e = tok["end_char"]
        if not isinstance(s, int) or not isinstance(e, int):
            raise PilotValidationError("start_char/end_char must be integers")
        if s < 0 or e < 0 or e < s or e > len(text):
            raise PilotValidationError("invalid start_char/end_char range")
        if text[s:e] != tok["text"]:
            raise PilotValidationError("token text does not match slice of text_ar")


def load_jsonl(path: str) -> List[Dict]:
    out: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def validate_jsonl(path: str) -> Tuple[int, List[str]]:
    selections = load_jsonl(path)
    refs: List[str] = []
    for s in selections:
        validate_selection(s)
        refs.append(s["reference"])
    return len(selections), refs


