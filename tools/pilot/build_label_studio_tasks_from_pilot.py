"""
Build Label Studio tasks JSONL from pilot selections JSONL.

Contract:
- Input: pilot selections JSONL (one selection per line)
- Output: Label Studio task JSONL where each line is a task dict with:
  - id, surah, ayah, raw_text_ar, tokens[], quran_text_version, tokenization_id
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Iterable, List


def _read_jsonl(path: str) -> List[Dict]:
    out: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def build_tasks(
    selections: List[Dict],
    quran_text_version: str,
    tokenization_id: str,
    id_prefix: str = "PILOT",
) -> List[Dict]:
    tasks: List[Dict] = []
    for i, sel in enumerate(selections, start=1):
        tid = f"{id_prefix}_{i:04d}"
        tasks.append(
            {
                "id": tid,
                "surah": sel["surah_number"],
                "ayah": sel["ayah_number"],
                "raw_text_ar": sel["text_ar"],
                "tokens": sel["tokens"],
                "quran_text_version": quran_text_version,
                "tokenization_id": tokenization_id,
            }
        )
    return tasks


def write_tasks_jsonl(out_path: str, tasks: List[Dict]) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False))
            f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True, help="Input pilot selections JSONL path.")
    parser.add_argument("--out", dest="out_path", required=True, help="Output Label Studio tasks JSONL path.")
    parser.add_argument("--quran-text-version", default="uthmani_hafs_v1")
    parser.add_argument("--tokenization-id", default="tok_v1")
    parser.add_argument("--id-prefix", default="PILOT")
    args = parser.parse_args()

    selections = _read_jsonl(args.in_path)
    tasks = build_tasks(selections, args.quran_text_version, args.tokenization_id, id_prefix=args.id_prefix)
    write_tasks_jsonl(args.out_path, tasks)
    print(f"Wrote {len(tasks)} tasks -> {args.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


