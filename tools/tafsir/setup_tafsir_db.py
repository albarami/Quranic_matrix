#!/usr/bin/env python3
"""
Set up SQLite database for tafsir data.

This creates a local SQLite database for fast tafsir lookups during annotation.
"""

import argparse
import json
import sqlite3
from pathlib import Path

TAFSIR_DIR = Path(__file__).parent.parent.parent / "data" / "tafsir"
DB_PATH = TAFSIR_DIR / "tafsir.db"


def create_schema(conn: sqlite3.Connection):
    """Create database tables."""
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tafsir_sources (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            language TEXT DEFAULT 'ar',
            ayat_count INTEGER DEFAULT 0
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tafsir_content (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_id TEXT NOT NULL,
            surah INTEGER NOT NULL,
            ayah INTEGER NOT NULL,
            text_ar TEXT,
            text_en TEXT,
            FOREIGN KEY (source_id) REFERENCES tafsir_sources(id),
            UNIQUE(source_id, surah, ayah)
        )
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_tafsir_ref 
        ON tafsir_content(surah, ayah)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_tafsir_source 
        ON tafsir_content(source_id)
    """)
    
    # Full-text search table
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS tafsir_fts 
        USING fts5(source_id, surah, ayah, text_ar, content=tafsir_content, content_rowid=id)
    """)
    
    conn.commit()
    print("Database schema created")


def load_tafsir_file(conn: sqlite3.Connection, filepath: Path, source_id: str):
    """Load a tafsir JSONL file into the database."""
    cursor = conn.cursor()
    
    # Get source name from first record
    source_name = source_id
    count = 0
    
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            
            record = json.loads(line)
            ref = record["reference"]
            
            if count == 0:
                source_name = record.get("resource_name", source_id)
            
            cursor.execute("""
                INSERT OR REPLACE INTO tafsir_content 
                (source_id, surah, ayah, text_ar)
                VALUES (?, ?, ?, ?)
            """, (source_id, ref["surah"], ref["ayah"], record.get("text_ar", "")))
            
            count += 1
    
    # Update source info
    cursor.execute("""
        INSERT OR REPLACE INTO tafsir_sources (id, name, language, ayat_count)
        VALUES (?, ?, 'ar', ?)
    """, (source_id, source_name, count))
    
    conn.commit()
    print(f"Loaded {count} records from {source_id}")
    return count


def rebuild_fts(conn: sqlite3.Connection):
    """Rebuild full-text search index."""
    cursor = conn.cursor()
    cursor.execute("INSERT INTO tafsir_fts(tafsir_fts) VALUES('rebuild')")
    conn.commit()
    print("FTS index rebuilt")


def show_stats(conn: sqlite3.Connection):
    """Show database statistics."""
    cursor = conn.cursor()
    
    print("\n" + "=" * 50)
    print("TAFSIR DATABASE STATISTICS")
    print("=" * 50)
    
    cursor.execute("SELECT id, name, ayat_count FROM tafsir_sources ORDER BY id")
    sources = cursor.fetchall()
    
    print(f"\nSources: {len(sources)}")
    for source_id, name, count in sources:
        print(f"  {source_id}: {name} ({count} ayat)")
    
    cursor.execute("SELECT COUNT(*) FROM tafsir_content")
    total = cursor.fetchone()[0]
    print(f"\nTotal records: {total}")
    
    # Coverage check
    cursor.execute("""
        SELECT surah, COUNT(DISTINCT ayah) as ayat_count 
        FROM tafsir_content 
        GROUP BY surah 
        ORDER BY surah
    """)
    coverage = cursor.fetchall()
    print(f"\nSurah coverage: {len(coverage)}/114")


def main():
    parser = argparse.ArgumentParser(description="Set up tafsir database")
    parser.add_argument("--db", default=str(DB_PATH), help="Database path")
    parser.add_argument("--load", help="Load a specific JSONL file")
    parser.add_argument("--load-all", action="store_true", help="Load all available tafsir files")
    parser.add_argument("--rebuild-fts", action="store_true", help="Rebuild FTS index")
    parser.add_argument("--stats", action="store_true", help="Show database statistics")
    args = parser.parse_args()
    
    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(db_path))
    create_schema(conn)
    
    if args.load:
        filepath = Path(args.load)
        source_id = filepath.stem.replace(".ar", "")
        load_tafsir_file(conn, filepath, source_id)
    
    if args.load_all:
        for jsonl_file in TAFSIR_DIR.glob("*.ar.jsonl"):
            source_id = jsonl_file.stem.replace(".ar", "")
            load_tafsir_file(conn, jsonl_file, source_id)
    
    if args.rebuild_fts:
        rebuild_fts(conn)
    
    if args.stats or args.load or args.load_all:
        show_stats(conn)
    
    conn.close()
    print(f"\nDatabase: {db_path}")


if __name__ == "__main__":
    main()
