"""
Clean all tafsir JSONL files in place.

This is the ROOT FIX for HTML contamination. Instead of maintaining
separate cleaned copies, we clean the source files directly so ALL
downstream code automatically uses clean data.

After running this:
- data/tafsir/*.ar.jsonl will have clean text_ar fields
- All code using CrossTafsirAnalyzer will get clean data
- Behavioral annotations extraction will produce clean data
- Embedding training will use clean data
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.text_cleaner import TextCleaner

PROJECT_ROOT = Path(__file__).parent.parent
TAFSIR_DIR = PROJECT_ROOT / "data" / "tafsir"
BACKUP_DIR = PROJECT_ROOT / "data" / "tafsir_backup"

TAFSIR_FILES = [
    "ibn_kathir.ar.jsonl",
    "tabari.ar.jsonl",
    "qurtubi.ar.jsonl",
    "saadi.ar.jsonl",
    "jalalayn.ar.jsonl",
]


def clean_tafsir_jsonl_files():
    """Clean all tafsir JSONL files in place."""
    
    cleaner = TextCleaner()
    
    # Create backup directory
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    total_stats = {
        "files_processed": 0,
        "records_processed": 0,
        "records_cleaned": 0,
        "html_chars_removed": 0,
    }
    
    for filename in TAFSIR_FILES:
        filepath = TAFSIR_DIR / filename
        if not filepath.exists():
            print(f"  Skipping {filename} (not found)")
            continue
        
        print(f"\nProcessing {filename}...")
        
        # Backup original
        backup_path = BACKUP_DIR / f"{filename}.{timestamp}.bak"
        shutil.copy(filepath, backup_path)
        print(f"  Backed up to {backup_path.name}")
        
        # Read, clean, and write
        cleaned_records = []
        file_stats = {"total": 0, "cleaned": 0, "html_removed": 0}
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                
                record = json.loads(line)
                file_stats["total"] += 1
                
                text = record.get("text_ar", "")
                if text and cleaner.has_html(text):
                    original_len = len(text)
                    cleaned_text = cleaner.clean(text)
                    record["text_ar"] = cleaned_text
                    file_stats["cleaned"] += 1
                    file_stats["html_removed"] += original_len - len(cleaned_text)
                
                cleaned_records.append(record)
        
        # Write cleaned data back
        with open(filepath, 'w', encoding='utf-8') as f:
            for record in cleaned_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        print(f"  Records: {file_stats['total']}")
        print(f"  Cleaned: {file_stats['cleaned']} ({file_stats['cleaned']/file_stats['total']*100:.1f}%)")
        print(f"  HTML chars removed: {file_stats['html_removed']:,}")
        
        total_stats["files_processed"] += 1
        total_stats["records_processed"] += file_stats["total"]
        total_stats["records_cleaned"] += file_stats["cleaned"]
        total_stats["html_chars_removed"] += file_stats["html_removed"]
    
    print("\n" + "="*60)
    print("CLEANING COMPLETE")
    print("="*60)
    print(f"Files processed: {total_stats['files_processed']}")
    print(f"Records processed: {total_stats['records_processed']:,}")
    print(f"Records cleaned: {total_stats['records_cleaned']:,}")
    print(f"HTML chars removed: {total_stats['html_chars_removed']:,}")
    print(f"\nBackups saved to: {BACKUP_DIR}")
    
    return total_stats


def verify_clean():
    """Verify all JSONL files are now clean."""
    cleaner = TextCleaner()
    
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    
    all_clean = True
    for filename in TAFSIR_FILES:
        filepath = TAFSIR_DIR / filename
        if not filepath.exists():
            continue
        
        contaminated = 0
        total = 0
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    total += 1
                    if cleaner.has_html(record.get("text_ar", "")):
                        contaminated += 1
        
        status = "✅ CLEAN" if contaminated == 0 else f"❌ {contaminated} contaminated"
        print(f"  {filename}: {status}")
        
        if contaminated > 0:
            all_clean = False
    
    return all_clean


if __name__ == "__main__":
    print("="*60)
    print("TAFSIR JSONL IN-PLACE CLEANING")
    print("="*60)
    print("\nThis will clean HTML from all tafsir JSONL files.")
    print("Original files will be backed up.\n")
    
    clean_tafsir_jsonl_files()
    
    if verify_clean():
        print("\n✅ All tafsir files are now clean!")
        print("All downstream code will automatically use clean data.")
    else:
        print("\n❌ Some files still have contamination. Check logs.")
