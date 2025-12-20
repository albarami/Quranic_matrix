"""Enable legacy API tokens in Label Studio database."""
import sqlite3
import os

db_path = os.path.join(os.environ['LOCALAPPDATA'], 'label-studio', 'label-studio', 'label_studio.sqlite3')
print(f"Database path: {db_path}")

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Enable legacy tokens
cursor.execute("UPDATE jwt_auth_jwtsettings SET legacy_api_tokens_enabled = 1 WHERE organization_id = 1")
conn.commit()

# Verify
cursor.execute("SELECT * FROM jwt_auth_jwtsettings")
cols = [d[0] for d in cursor.description]
print(f"Columns: {cols}")
for row in cursor.fetchall():
    print(row)

print("\nLegacy API tokens enabled!")
conn.close()
