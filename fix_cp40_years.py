#!/usr/bin/env python3
"""
Fix year extraction in existing CP40 database entries.

This script corrects years that were incorrectly extracted from CP40/xxxx
to properly extract from (Term yyyy) pattern in the raw_text, where Term is
one of: Hilary, Easter, Trinity, or Michaelmas.
"""

import sqlite3
import re
import argparse
from typing import Optional


def extract_year_from_term(raw_text: str) -> Optional[int]:
    """
    Extract year from (Term yyyy) pattern in raw_text.
    Term can be: Hilary, Easter, Trinity, or Michaelmas.
    
    Args:
        raw_text: The raw text entry from the database
        
    Returns:
        Year as integer if found, None otherwise
    """
    if not raw_text:
        return None
    
    # Look for pattern like "(Hilary 1400)" or "(Michaelmas 1450)" or "(Easter: 1400)"
    # Legal terms: Hilary, Easter, Trinity, Michaelmas
    # Pattern allows for optional colon and whitespace variations
    year_match = re.search(
        r'\((?:Hilary|Easter|Trinity|Michaelmas)\s*:?\s*(\d{4})\)', 
        raw_text, 
        re.IGNORECASE
    )
    if year_match:
        try:
            return int(year_match.group(1))
        except ValueError:
            return None
    
    return None


def fix_database_years(db_path: str, dry_run: bool = False) -> dict:
    """
    Fix year extraction in all database entries.
    
    Args:
        db_path: Path to SQLite database
        dry_run: If True, don't actually update the database
        
    Returns:
        Dictionary with diagnostic information and counts
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    # Get all entries (including roll_reference for potential year extraction)
    cursor = conn.execute("SELECT id, raw_text, roll_reference, year FROM entries")
    entries = cursor.fetchall()
    
    entries_updated = 0
    entries_checked = len(entries)
    entries_with_term_pattern = 0
    entries_with_cp40_pattern = 0
    entries_with_year_mismatch = 0
    sample_mismatches = []
    
    print(f"Checking {entries_checked} entries...")
    
    for entry in entries:
        entry_id = entry['id']
        raw_text = entry['raw_text']
        try:
            roll_reference = entry['roll_reference']
        except (KeyError, IndexError):
            roll_reference = ''
        current_year = entry['year']
        
        # Check for patterns for diagnostics
        if re.search(r'\((?:Hilary|Easter|Trinity|Michaelmas)\s*:?\s*\d{4}\)', raw_text, re.IGNORECASE):
            entries_with_term_pattern += 1
        if re.search(r'CP40/\d+', raw_text):
            entries_with_cp40_pattern += 1
        
        # Extract year from (term yyyy) pattern - check both raw_text and roll_reference
        correct_year = extract_year_from_term(raw_text)
        if correct_year is None and roll_reference:
            correct_year = extract_year_from_term(roll_reference)
        
        if correct_year is not None:
            # Check if year needs to be updated
            if current_year != correct_year:
                entries_with_year_mismatch += 1
                if len(sample_mismatches) < 5:
                    sample_mismatches.append({
                        'id': entry_id,
                        'current': current_year,
                        'correct': correct_year,
                        'raw_preview': raw_text[:200]
                    })
                
                if dry_run:
                    if len(sample_mismatches) <= 5:
                        print(f"  Entry {entry_id}: Would update year from {current_year} to {correct_year}")
                else:
                    conn.execute(
                        "UPDATE entries SET year = ? WHERE id = ?",
                        (correct_year, entry_id)
                    )
                    entries_updated += 1
            elif current_year is None:
                # Year was missing, now we can set it
                if dry_run:
                    if entries_updated < 5:
                        print(f"  Entry {entry_id}: Would set year to {correct_year}")
                else:
                    conn.execute(
                        "UPDATE entries SET year = ? WHERE id = ?",
                        (correct_year, entry_id)
                    )
                    entries_updated += 1
    
    if not dry_run:
        conn.commit()
    
    conn.close()
    
    return {
        'entries_updated': entries_updated,
        'entries_checked': entries_checked,
        'entries_with_term_pattern': entries_with_term_pattern,
        'entries_with_cp40_pattern': entries_with_cp40_pattern,
        'entries_with_year_mismatch': entries_with_year_mismatch,
        'sample_mismatches': sample_mismatches
    }


def main():
    parser = argparse.ArgumentParser(
        description='Fix year extraction in CP40 database entries',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to see what would be changed
  python fix_cp40_years.py --db cp40_records.db --dry-run
  
  # Actually fix the database
  python fix_cp40_years.py --db cp40_records.db
        """
    )
    
    parser.add_argument(
        '--db',
        default='cp40_records.db',
        help='Path to SQLite database (default: cp40_records.db)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be changed without actually updating the database'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CP40 YEAR CORRECTION SCRIPT")
    print("=" * 60)
    print(f"Database: {args.db}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'UPDATE'}")
    print()
    
    try:
        results = fix_database_years(
            args.db, 
            dry_run=args.dry_run
        )
        
        entries_updated = results['entries_updated']
        entries_checked = results['entries_checked']
        entries_with_term_pattern = results['entries_with_term_pattern']
        entries_with_cp40_pattern = results['entries_with_cp40_pattern']
        entries_with_year_mismatch = results['entries_with_year_mismatch']
        sample_mismatches = results['sample_mismatches']
        
        print()
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Entries checked: {entries_checked:,}")
        print(f"Entries with '(term yyyy)' pattern: {entries_with_term_pattern:,}")
        print(f"Entries with 'CP40/xxxx' pattern: {entries_with_cp40_pattern:,}")
        print(f"Entries with year mismatch: {entries_with_year_mismatch:,}")
        print(f"Entries {'would be ' if args.dry_run else ''}updated: {entries_updated:,}")
        
        if sample_mismatches:
            print("\nSample entries that would be updated:")
            for sample in sample_mismatches[:5]:
                print(f"  Entry {sample['id']}: year {sample['current']} -> {sample['correct']}")
                print(f"    Raw text preview: {sample['raw_preview']}...")
        
        if args.dry_run and entries_updated > 0:
            print("\nRun without --dry-run to apply these changes.")
        elif entries_updated == 0 and entries_with_term_pattern == 0:
            print("\n⚠️  WARNING: No entries found with '(Term yyyy)' pattern.")
            print("   Expected patterns: (Hilary 1400), (Easter 1400), (Trinity 1400), (Michaelmas 1400)")
            print("   This might indicate the data format is different than expected.")
            print("   Consider running inspect_cp40_format.py to examine the data format.")
        
        return 0
        
    except FileNotFoundError:
        print(f"Error: Database file not found: {args.db}")
        return 1
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())

