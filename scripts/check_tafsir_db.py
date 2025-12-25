"""Quick script to check tafsir database structure"""
import sqlite3

conn = sqlite3.connect(r'data/tafsir/tafsir.db')
cursor = conn.cursor()

# Get tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in cursor.fetchall()]
print("Tables:", tables)

# Check first table structure
if tables:
    for table in tables[:3]:
        cursor.execute(f"PRAGMA table_info({table})")
        cols = cursor.fetchall()
        print(f"\n{table} columns:", [c[1] for c in cols])
        
        cursor.execute(f"SELECT * FROM {table} LIMIT 1")
        row = cursor.fetchone()
        if row:
            print(f"Sample row: {str(row)[:200]}...")

conn.close()
