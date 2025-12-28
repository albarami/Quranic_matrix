"""Clean and deduplicate newly downloaded tafsir files."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.preprocessing.text_cleaner import TextCleaner

def clean_and_dedupe(filename: str):
    """Clean HTML and remove duplicates from a tafsir file."""
    cleaner = TextCleaner()
    filepath = Path("data/tafsir") / filename
    
    if not filepath.exists():
        print(f"File not found: {filepath}")
        return
    
    # Read and deduplicate
    seen = set()
    records = []
    duplicates = 0
    html_cleaned = 0
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                ref = record.get('reference', {})
                key = f"{ref.get('surah')}:{ref.get('ayah')}"
                
                if key in seen:
                    duplicates += 1
                    continue
                seen.add(key)
                
                # Clean HTML
                text = record.get('text_ar', '')
                if text and cleaner.has_html(text):
                    record['text_ar'] = cleaner.clean(text)
                    html_cleaned += 1
                
                records.append(record)
    
    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"{filename}:")
    print(f"  Records: {len(records)}")
    print(f"  Duplicates removed: {duplicates}")
    print(f"  HTML cleaned: {html_cleaned}")
    return len(records)


if __name__ == "__main__":
    files_to_clean = ["muyassar.ar.jsonl", "baghawi.ar.jsonl"]
    
    for f in files_to_clean:
        clean_and_dedupe(f)
