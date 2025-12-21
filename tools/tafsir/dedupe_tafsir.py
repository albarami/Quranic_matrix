#!/usr/bin/env python3
"""
Deduplicate tafsir JSONL file.
Keeps the record with longer text_ar for each unique reference.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict


def analyze_duplicates(filepath: Path):
    """Analyze duplicates in tafsir file."""
    records = defaultdict(list)
    
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                rec = json.loads(line)
                ref = rec["reference"]
                key = f"{ref['surah']}:{ref['ayah']}"
                records[key].append((i, rec))
    
    total_lines = sum(len(v) for v in records.values())
    unique_refs = len(records)
    dups = {k: v for k, v in records.items() if len(v) > 1}
    
    print(f"Total lines: {total_lines}")
    print(f"Unique refs: {unique_refs}")
    print(f"Duplicated refs: {len(dups)}")
    
    if dups:
        # Analyze sources
        api_wins = 0
        cdn_wins = 0
        
        for key, entries in dups.items():
            best = max(entries, key=lambda x: len(x[1].get("text_ar", "")))
            has_rn = "resource_name" in best[1]
            if has_rn:
                api_wins += 1
            else:
                cdn_wins += 1
        
        print(f"\nWhen keeping longer text:")
        print(f"  API records win: {api_wins}")
        print(f"  CDN records win: {cdn_wins}")
        
        # Show example
        k = list(dups.keys())[0]
        print(f"\nExample duplicate: {k}")
        for i, rec in dups[k]:
            has_rn = "resource_name" in rec
            text_len = len(rec.get("text_ar", ""))
            print(f"  Line {i}: resource_name={has_rn}, text_len={text_len}")
    
    return records


def dedupe_file(filepath: Path, output: Path = None):
    """Deduplicate file, keeping longer text_ar for each ref."""
    records = defaultdict(list)
    
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                ref = rec["reference"]
                key = f"{ref['surah']}:{ref['ayah']}"
                records[key].append(rec)
    
    # Keep record with longest text_ar
    deduped = {}
    for key, entries in records.items():
        best = max(entries, key=lambda x: len(x.get("text_ar", "")))
        # Normalize: ensure consistent fields
        ref_parts = key.split(":")
        deduped[key] = {
            "tafsir_id": "ibn_kathir",
            "reference": {"surah": int(ref_parts[0]), "ayah": int(ref_parts[1])},
            "text_ar": best.get("text_ar", ""),
        }
        if "resource_name" in best:
            deduped[key]["resource_name"] = best["resource_name"]
    
    # Sort by surah:ayah
    sorted_keys = sorted(deduped.keys(), key=lambda x: (int(x.split(":")[0]), int(x.split(":")[1])))
    
    output = output or filepath
    with open(output, "w", encoding="utf-8") as f:
        for key in sorted_keys:
            f.write(json.dumps(deduped[key], ensure_ascii=False) + "\n")
    
    print(f"Deduped: {len(sorted_keys)} unique records written to {output}")
    return len(sorted_keys)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="JSONL file to dedupe")
    parser.add_argument("--analyze", action="store_true", help="Only analyze, don't modify")
    parser.add_argument("--output", help="Output file (default: overwrite input)")
    args = parser.parse_args()
    
    filepath = Path(args.file)
    
    if args.analyze:
        analyze_duplicates(filepath)
    else:
        output = Path(args.output) if args.output else filepath
        dedupe_file(filepath, output)
