"""Check QBM database coverage."""
import json
from pathlib import Path

DATA_FILE = Path(__file__).parent.parent.parent / "data" / "exports" / "qbm_silver_20251221.json"

with open(DATA_FILE, encoding="utf-8") as f:
    data = json.load(f)

spans = data["spans"]
unique_ayat = set()

for s in spans:
    ref = s.get("reference", {})
    key = f"{ref.get('surah')}:{ref.get('ayah')}"
    unique_ayat.add(key)

print(f"Total spans: {len(spans)}")
print(f"Unique ayat covered: {len(unique_ayat)}")
print(f"Total Quran ayat: 6236")
print(f"Coverage: {len(unique_ayat)/6236*100:.1f}%")

# Check if spans == ayat (1:1 mapping)
if len(spans) == len(unique_ayat):
    print("\nNote: Each span corresponds to exactly one ayah (1:1 mapping)")
else:
    print(f"\nNote: Some ayat have multiple spans ({len(spans) - len(unique_ayat)} extra)")
