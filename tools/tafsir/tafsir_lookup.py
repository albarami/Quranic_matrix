#!/usr/bin/env python3
"""
Tafsir lookup tool for QBM annotation workflow.

Usage:
    python tafsir_lookup.py --surah 2 --ayah 255
    python tafsir_lookup.py --surah 2 --ayah 255 --source ibn_kathir
    python tafsir_lookup.py --surah 2 --ayah 255 --compare
    python tafsir_lookup.py --search "الكرسي"
"""

import argparse
import json
import re
from pathlib import Path
from typing import Optional, List, Dict

# Default tafsir directory
TAFSIR_DIR = Path(__file__).parent.parent.parent / "data" / "tafsir"

# Available sources (priority order)
TAFSIR_SOURCES = ["ibn_kathir", "tabari", "qurtubi", "saadi", "muyassar", "baghawi"]


def load_tafsir_index(tafsir_dir: Path) -> Dict[str, Dict]:
    """Load all tafsir files into memory index."""
    index = {}
    
    for source in TAFSIR_SOURCES:
        filepath = tafsir_dir / f"{source}.ar.jsonl"
        if not filepath.exists():
            continue
        
        index[source] = {}
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    ref = record["reference"]
                    key = f"{ref['surah']}:{ref['ayah']}"
                    index[source][key] = record
    
    return index


def strip_html(text: str) -> str:
    """Remove HTML tags from tafsir text."""
    clean = re.sub(r'<[^>]+>', '', text)
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean


def get_tafsir(index: Dict, surah: int, ayah: int, source: Optional[str] = None) -> Dict:
    """Get tafsir for a specific ayah."""
    key = f"{surah}:{ayah}"
    
    if source:
        if source not in index:
            return {"error": f"Source '{source}' not available"}
        if key not in index[source]:
            return {"error": f"No tafsir found for {key} in {source}"}
        
        record = index[source][key]
        return {
            "source": source,
            "reference": f"{surah}:{ayah}",
            "text": strip_html(record.get("text_ar", "")),
            "resource_name": record.get("resource_name", source)
        }
    
    # Return first available source
    for src in TAFSIR_SOURCES:
        if src in index and key in index[src]:
            record = index[src][key]
            return {
                "source": src,
                "reference": f"{surah}:{ayah}",
                "text": strip_html(record.get("text_ar", "")),
                "resource_name": record.get("resource_name", src)
            }
    
    return {"error": f"No tafsir found for {key}"}


def compare_tafsirs(index: Dict, surah: int, ayah: int) -> List[Dict]:
    """Get tafsir from all available sources for comparison."""
    key = f"{surah}:{ayah}"
    results = []
    
    for source in TAFSIR_SOURCES:
        if source in index and key in index[source]:
            record = index[source][key]
            results.append({
                "source": source,
                "resource_name": record.get("resource_name", source),
                "text": strip_html(record.get("text_ar", ""))[:500] + "..."
            })
    
    return results


def search_tafsir(index: Dict, query: str, source: Optional[str] = None, limit: int = 10) -> List[Dict]:
    """Search for a term across tafsir texts."""
    results = []
    sources_to_search = [source] if source else TAFSIR_SOURCES
    
    for src in sources_to_search:
        if src not in index:
            continue
        
        for key, record in index[src].items():
            text = record.get("text_ar", "")
            if query in text:
                results.append({
                    "source": src,
                    "reference": key,
                    "snippet": extract_snippet(text, query, 100)
                })
                
                if len(results) >= limit:
                    return results
    
    return results


def extract_snippet(text: str, query: str, context: int = 100) -> str:
    """Extract a snippet around the query term."""
    text = strip_html(text)
    pos = text.find(query)
    if pos == -1:
        return ""
    
    start = max(0, pos - context)
    end = min(len(text), pos + len(query) + context)
    
    snippet = text[start:end]
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet = snippet + "..."
    
    return snippet


def main():
    parser = argparse.ArgumentParser(description="Tafsir lookup tool")
    parser.add_argument("--surah", type=int, help="Surah number (1-114)")
    parser.add_argument("--ayah", type=int, help="Ayah number")
    parser.add_argument("--end-ayah", type=int, help="End ayah for range lookup")
    parser.add_argument("--source", choices=TAFSIR_SOURCES, help="Specific tafsir source")
    parser.add_argument("--compare", action="store_true", help="Compare all available sources")
    parser.add_argument("--search", help="Search term in tafsir texts")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--tafsir-dir", default=str(TAFSIR_DIR), help="Tafsir data directory")
    args = parser.parse_args()
    
    tafsir_dir = Path(args.tafsir_dir)
    
    # Check available sources
    available = [s for s in TAFSIR_SOURCES if (tafsir_dir / f"{s}.ar.jsonl").exists()]
    if not available:
        print(f"No tafsir files found in {tafsir_dir}")
        print("Run: python tools/tafsir/download_tafsir.py --tafsir ibn_kathir")
        return
    
    print(f"Available sources: {', '.join(available)}")
    
    # Load index
    print("Loading tafsir index...")
    index = load_tafsir_index(tafsir_dir)
    print(f"Loaded {sum(len(v) for v in index.values())} records from {len(index)} sources")
    
    # Handle search
    if args.search:
        results = search_tafsir(index, args.search, args.source)
        if args.json:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            print(f"\nSearch results for '{args.search}':")
            for r in results:
                print(f"\n[{r['source']}] {r['reference']}")
                print(f"  {r['snippet']}")
        return
    
    # Handle ayah lookup
    if args.surah and args.ayah:
        end_ayah = args.end_ayah or args.ayah
        
        for ayah in range(args.ayah, end_ayah + 1):
            if args.compare:
                results = compare_tafsirs(index, args.surah, ayah)
                if args.json:
                    print(json.dumps(results, ensure_ascii=False, indent=2))
                else:
                    print(f"\n{'='*60}")
                    print(f"Tafsir comparison for {args.surah}:{ayah}")
                    print(f"{'='*60}")
                    for r in results:
                        print(f"\n[{r['resource_name']}]")
                        print(r['text'])
            else:
                result = get_tafsir(index, args.surah, ayah, args.source)
                if args.json:
                    print(json.dumps(result, ensure_ascii=False, indent=2))
                else:
                    if "error" in result:
                        print(f"Error: {result['error']}")
                    else:
                        print(f"\n{'='*60}")
                        print(f"[{result['resource_name']}] {result['reference']}")
                        print(f"{'='*60}")
                        print(result['text'])
        return
    
    parser.print_help()


if __name__ == "__main__":
    main()
