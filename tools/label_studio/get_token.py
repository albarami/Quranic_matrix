"""Get API token from Label Studio database."""
import sqlite3
import os

db_path = os.path.join(os.environ['LOCALAPPDATA'], 'label-studio', 'label-studio', 'label_studio.sqlite3')
print(f"Database path: {db_path}")

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# List tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in cursor.fetchall()]
print(f"Tables: {tables}")

# Find auth token table
if 'authtoken_token' in tables:
    cursor.execute("SELECT key, user_id FROM authtoken_token")
    tokens = cursor.fetchall()
    for token, user_id in tokens:
        print(f"Token for user {user_id}: {token}")

conn.close()
