import sqlite3
import os

DB_FILE = "CP40_Migrated.db"

def print_full_schema():
    if not os.path.exists(DB_FILE):
        print(f"Error: Database file '{DB_FILE}' not found.")
        return

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    print(f"--- Schema for {DB_FILE} ---\n")

    # 1. Get a list of all tables in the database
    # We filter out 'sqlite_sequence' which is an internal system table
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;")
    tables = cursor.fetchall()

    if not tables:
        print("No tables found in the database.")
        conn.close()
        return

    # 2. Iterate through each table to get column details
    for table_row in tables:
        table_name = table_row[0]
        
        print(f"Table: {table_name}")
        
        # PRAGMA table_info returns: (cid, name, type, notnull, dflt_value, pk)
        # We need to wrap table_name in quotes in case it contains spaces or symbols
        cursor.execute(f'PRAGMA table_info("{table_name}")')
        columns = cursor.fetchall()
        
        if not columns:
            print("  [No columns found]")
        
        for col in columns:
            col_name = col[1]
            col_type = col[2]
            # Optional: Check if it's a primary key
            pk_marker = " [PK]" if col[5] == 1 else ""
            
            print(f"  - {col_name} ({col_type}){pk_marker}")
        
        print("-" * 30)

    conn.close()

if __name__ == "__main__":
    print_full_schema()