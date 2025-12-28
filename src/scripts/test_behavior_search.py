"""Test behavior search in QBM database."""
import json
from pathlib import Path

DATA_FILE = Path(__file__).parent.parent.parent / "data" / "exports" / "qbm_silver_20251221.json"

with open(DATA_FILE, encoding="utf-8") as f:
    data = json.load(f)

spans = data["spans"]

# Search for patience-related terms
search_terms = ["صبر", "الصبر", "صابر", "اصبر", "الصابرين", "صبرا"]
matches = []

for span in spans:
    text = span.get("text_ar", "")
    for term in search_terms:
        if term in text:
            matches.append(span)
            break

print(f"Found {len(matches)} matches for patience-related terms")
print()

for m in matches[:10]:
    ref = m["reference"]
    print(f"[{ref['surah']}:{ref['ayah']}] {ref.get('surah_name', '')}")
    print(f"  Text: {m['text_ar'][:80]}...")
    print(f"  Agent: {m.get('agent', {}).get('type')}")
    print(f"  Form: {m.get('behavior_form')}")
    print(f"  Eval: {m.get('normative', {}).get('evaluation')}")
    print()
