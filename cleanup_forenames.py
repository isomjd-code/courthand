#!/usr/bin/env python3
"""
Clean up garbage forename entries from cp40_records.db.

These are placeholder names like R_, W_, J_, R--, etc. that were extracted
from damaged or incomplete records. When latinized, Gemini incorrectly
guessed what names they might be, creating conflicting Latin forms.
"""
import sqlite3
import argparse

def find_garbage_forenames(conn):
    """Find forenames that look like placeholders/garbage."""
    cursor = conn.execute('''
        SELECT id, english_name, frequency
        FROM forenames 
        WHERE 
            -- Contains underscore or multiple dashes
            english_name GLOB '*[_]*'
            OR english_name GLOB '*--*'
            -- Single letter followed by dash
            OR english_name GLOB '[A-Z]-'
            -- Just dashes
            OR english_name GLOB '-*'
            -- Very short non-names (but exclude valid short names)
            OR (LENGTH(english_name) <= 2 
                AND english_name NOT IN ('Al', 'Ed', 'Jo', 'Ro'))
        ORDER BY frequency DESC
    ''')
    return cursor.fetchall()


def delete_forename(conn, forename_id):
    """Delete a forename and all its related data."""
    # Delete Latin forms
    conn.execute('DELETE FROM forename_latin_forms WHERE forename_id = ?', (forename_id,))
    # Delete processing job
    conn.execute('DELETE FROM forename_processing_jobs WHERE forename_id = ?', (forename_id,))
    # Delete forename
    conn.execute('DELETE FROM forenames WHERE id = ?', (forename_id,))


def main():
    parser = argparse.ArgumentParser(description='Clean up garbage forename entries')
    parser.add_argument('--db', default='cp40_records.db', help='Database path')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be deleted without deleting')
    parser.add_argument('--yes', action='store_true', help='Skip confirmation prompt')
    args = parser.parse_args()
    
    conn = sqlite3.connect(args.db)
    
    garbage = find_garbage_forenames(conn)
    
    if not garbage:
        print("No garbage forenames found!")
        return
    
    print(f"Found {len(garbage)} garbage forenames:\n")
    print(f"{'ID':>6}  {'Name':<15}  {'Freq':>6}")
    print("-" * 35)
    for fid, name, freq in garbage:
        print(f"{fid:>6}  {name:<15}  {freq:>6}")
    
    # Count associated Latin forms
    total_latin_forms = 0
    for fid, _, _ in garbage:
        cursor = conn.execute(
            'SELECT COUNT(*) FROM forename_latin_forms WHERE forename_id = ?', 
            (fid,)
        )
        total_latin_forms += cursor.fetchone()[0]
    
    print(f"\nThis will also delete {total_latin_forms} associated Latin forms.")
    
    if args.dry_run:
        print("\n[DRY RUN] No changes made.")
        return
    
    if not args.yes:
        response = input("\nDelete these entries? [y/N] ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Delete
    for fid, name, freq in garbage:
        delete_forename(conn, fid)
        print(f"  Deleted: {name}")
    
    conn.commit()
    conn.close()
    
    print(f"\n✓ Deleted {len(garbage)} garbage forenames and {total_latin_forms} Latin forms.")


if __name__ == '__main__':
    main()

