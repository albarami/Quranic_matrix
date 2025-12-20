"""Get all tokens from Label Studio database."""
import sqlite3
import os

db_path = os.path.join(os.environ['LOCALAPPDATA'], 'label-studio', 'label-studio', 'label_studio.sqlite3')
print(f"Database path: {db_path}")

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Check token_blacklist_outstandingtoken for JWT tokens
print("\n--- Outstanding Tokens ---")
cursor.execute("SELECT * FROM token_blacklist_outstandingtoken LIMIT 5")
cols = [d[0] for d in cursor.description]
print(f"Columns: {cols}")
for row in cursor.fetchall():
    print(row)

# Check htx_user for user info
print("\n--- Users ---")
cursor.execute("SELECT id, email, username FROM htx_user")
for row in cursor.fetchall():
    print(row)

# Check if there's a way to enable legacy tokens or get JWT
print("\n--- JWT Settings ---")
cursor.execute("SELECT * FROM jwt_auth_jwtsettings")
cols = [d[0] for d in cursor.description]
print(f"Columns: {cols}")
for row in cursor.fetchall():
    print(row)

conn.close()
