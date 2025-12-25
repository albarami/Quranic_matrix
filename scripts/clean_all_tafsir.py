"""
QBM Tafsir Cleaning Script - Phase 3
Re-processes all tafsir data to remove HTML contamination.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.text_cleaner import TextCleaner, analyze_html_contamination
from src.preprocessing.provenance import CleanedTafsirRecord


def clean_tafsir_database(
    db_path: str = "data/tafsir/tafsir.db",
    output_path: str = "data/tafsir/tafsir_cleaned.db",
    report_path: str = "reports/tafsir_cleaning_report.json"
):
    """
    Clean all tafsir content in the database.
    
    Args:
        db_path: Path to original tafsir database
        output_path: Path for cleaned database
        report_path: Path for cleaning report
    """
    cleaner = TextCleaner(
        strip_html=True,
        normalize_arabic=True,
        remove_diacritics=False,
        normalize_whitespace=True,
    )
    
    # Connect to source database
    src_conn = sqlite3.connect(db_path)
    src_cursor = src_conn.cursor()
    
    # Create output database
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    dst_conn = sqlite3.connect(output_path)
    dst_cursor = dst_conn.cursor()
    
    # Create cleaned table
    dst_cursor.execute("""
        CREATE TABLE IF NOT EXISTS tafsir_content_cleaned (
            id INTEGER PRIMARY KEY,
            source_id TEXT NOT NULL,
            surah INTEGER NOT NULL,
            ayah INTEGER NOT NULL,
            text_original TEXT,
            text_cleaned TEXT NOT NULL,
            html_contamination_rate REAL,
            cleaned_at TEXT,
            cleaner_version TEXT DEFAULT '1.0.0'
        )
    """)
    
    # Copy sources table
    dst_cursor.execute("""
        CREATE TABLE IF NOT EXISTS tafsir_sources (
            id TEXT PRIMARY KEY,
            name TEXT,
            language TEXT,
            ayat_count INTEGER
        )
    """)
    
    src_cursor.execute("SELECT * FROM tafsir_sources")
    for row in src_cursor.fetchall():
        dst_cursor.execute(
            "INSERT OR REPLACE INTO tafsir_sources VALUES (?, ?, ?, ?)",
            row
        )
    
    # Process all tafsir content
    src_cursor.execute("SELECT id, source_id, surah, ayah, text_ar FROM tafsir_content")
    
    stats = {
        "total_records": 0,
        "contaminated_records": 0,
        "total_html_chars_removed": 0,
        "by_source": {},
        "started_at": datetime.utcnow().isoformat(),
    }
    
    batch = []
    batch_size = 1000
    
    print("Cleaning tafsir content...")
    
    for row in src_cursor:
        record_id, source_id, surah, ayah, text_ar = row
        stats["total_records"] += 1
        
        if source_id not in stats["by_source"]:
            stats["by_source"][source_id] = {
                "total": 0,
                "contaminated": 0,
                "html_chars_removed": 0
            }
        
        stats["by_source"][source_id]["total"] += 1
        
        # Check for HTML contamination
        original_len = len(text_ar) if text_ar else 0
        has_html = cleaner.has_html(text_ar) if text_ar else False
        contamination_rate = cleaner.get_html_contamination_rate(text_ar) if text_ar else 0.0
        
        if has_html:
            stats["contaminated_records"] += 1
            stats["by_source"][source_id]["contaminated"] += 1
        
        # Clean the text
        text_cleaned = cleaner.clean(text_ar) if text_ar else ""
        
        html_chars = original_len - len(text_cleaned) if original_len > 0 else 0
        if html_chars > 0:
            stats["total_html_chars_removed"] += html_chars
            stats["by_source"][source_id]["html_chars_removed"] += html_chars
        
        batch.append((
            record_id,
            source_id,
            surah,
            ayah,
            text_ar,
            text_cleaned,
            contamination_rate,
            datetime.utcnow().isoformat(),
            "1.0.0"
        ))
        
        if len(batch) >= batch_size:
            dst_cursor.executemany(
                """INSERT OR REPLACE INTO tafsir_content_cleaned 
                   (id, source_id, surah, ayah, text_original, text_cleaned, 
                    html_contamination_rate, cleaned_at, cleaner_version)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                batch
            )
            dst_conn.commit()
            print(f"  Processed {stats['total_records']} records...")
            batch = []
    
    # Insert remaining batch
    if batch:
        dst_cursor.executemany(
            """INSERT OR REPLACE INTO tafsir_content_cleaned 
               (id, source_id, surah, ayah, text_original, text_cleaned, 
                html_contamination_rate, cleaned_at, cleaner_version)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            batch
        )
        dst_conn.commit()
    
    # Create FTS index on cleaned text
    dst_cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS tafsir_fts_cleaned USING fts5(
            text_cleaned,
            content='tafsir_content_cleaned',
            content_rowid='id'
        )
    """)
    
    dst_cursor.execute("""
        INSERT INTO tafsir_fts_cleaned(tafsir_fts_cleaned) VALUES('rebuild')
    """)
    
    dst_conn.commit()
    
    # Calculate final stats
    stats["completed_at"] = datetime.utcnow().isoformat()
    stats["pre_clean_contamination_rate"] = (
        stats["contaminated_records"] / stats["total_records"]
        if stats["total_records"] > 0 else 0.0
    )
    
    # Verify post-clean contamination rate
    dst_cursor.execute("SELECT text_cleaned FROM tafsir_content_cleaned LIMIT 1000")
    post_clean_contaminated = 0
    sample_size = 0
    for (text,) in dst_cursor.fetchall():
        sample_size += 1
        if cleaner.has_html(text):
            post_clean_contaminated += 1
    
    stats["post_clean_contamination_rate"] = (
        post_clean_contaminated / sample_size if sample_size > 0 else 0.0
    )
    stats["post_clean_sample_size"] = sample_size
    
    # Save report
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== Cleaning Complete ===")
    print(f"Total records: {stats['total_records']}")
    print(f"Contaminated records: {stats['contaminated_records']}")
    print(f"Pre-clean contamination rate: {stats['pre_clean_contamination_rate']:.2%}")
    print(f"Post-clean contamination rate: {stats['post_clean_contamination_rate']:.2%}")
    print(f"HTML chars removed: {stats['total_html_chars_removed']:,}")
    print(f"\nBy source:")
    for source, source_stats in stats["by_source"].items():
        rate = source_stats["contaminated"] / source_stats["total"] if source_stats["total"] > 0 else 0
        print(f"  {source}: {source_stats['contaminated']}/{source_stats['total']} ({rate:.1%})")
    print(f"\nReport saved to: {report_path}")
    print(f"Cleaned DB saved to: {output_path}")
    
    src_conn.close()
    dst_conn.close()
    
    return stats


if __name__ == "__main__":
    clean_tafsir_database()
