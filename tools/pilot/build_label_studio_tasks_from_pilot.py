"""
Build Label Studio tasks JSONL from pilot selections JSONL.

Contract:
- Input: pilot selections JSONL (one selection per line)
- Output: Label Studio task JSONL where each line is a task dict with:
  - id (integer, Label Studio task id for import)
  - data (object) containing the fields used by the labeling interface:
    - id (string, e.g. QBM_00001)
    - surah, surah_name, ayah, reference
    - raw_text_ar, token_count, tokens[]
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
    qbm_id_prefix: str = "QBM",
    qbm_id_width: int = 5,
    outer_id_start: int = 1,
) -> List[Dict]:
    tasks: List[Dict] = []
    for i, sel in enumerate(selections, start=outer_id_start):
        qbm_id = f"{qbm_id_prefix}_{i:0{qbm_id_width}d}"
        tasks.append(
            {
                "id": i,
                "data": {
                    "id": qbm_id,
                    "surah": sel["surah_number"],
                    "surah_name": sel.get("surah_name", ""),
                    "ayah": sel["ayah_number"],
                    "reference": sel.get("reference", f'{sel["surah_number"]}:{sel["ayah_number"]}'),
                    "raw_text_ar": sel["text_ar"],
                    "token_count": sel["token_count"],
                    "tokens": sel["tokens"],
                },
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
    parser.add_argument("--qbm-id-prefix", default="QBM")
    parser.add_argument("--qbm-id-width", type=int, default=5)
    parser.add_argument("--outer-id-start", type=int, default=1)
    args = parser.parse_args()

    selections = _read_jsonl(args.in_path)
    tasks = build_tasks(
        selections,
        qbm_id_prefix=args.qbm_id_prefix,
        qbm_id_width=args.qbm_id_width,
        outer_id_start=args.outer_id_start,
    )
    write_tasks_jsonl(args.out_path, tasks)
    print(f"Wrote {len(tasks)} tasks -> {args.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


