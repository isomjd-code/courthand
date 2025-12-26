#!/usr/bin/env python3
"""
SQLite Database module for CP40 scraper
Handles schema creation, duplicate prevention, and data storage
"""

import sqlite3
import hashlib
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime


# Available counties from the CP40 search form
COUNTIES = [
    "All",
    "Bedfordshire", "Berkshire", "Buckinghamshire", "Cambridgeshire",
    "Cheshire", "Cornwall", "Cumberland", "Derbyshire", "Devon",
    "Dorset", "Durham", "Essex", "Gloucestershire", "Hampshire",
    "Herefordshire", "Hertfordshire", "Huntingdonshire", "Kent",
    "Lancashire", "Leicestershire", "Lincolnshire", "London", "Middlesex",
    "Norfolk", "Northamptonshire", "Northumberland", "Nottinghamshire",
    "Oxfordshire", "Rutland", "Shropshire", "Somerset", "Staffordshire",
    "Suffolk", "Surrey", "Sussex", "Warwickshire", "Westmorland",
    "Wiltshire", "Worcestershire", "Yorkshire"
]

# Year range for CP40 records
YEAR_START = 1349
YEAR_END = 1596


SCHEMA_SQL = """
-- ═══════════════════════════════════════════════════════════════
-- MAIN ENTRIES TABLE
-- ═══════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    roll_reference TEXT NOT NULL,
    raw_text TEXT NOT NULL,
    raw_text_hash TEXT UNIQUE NOT NULL,
    county TEXT,
    year INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ═══════════════════════════════════════════════════════════════
-- NORMALIZED PERSONS (many-to-many with entries)
-- ═══════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS persons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS entry_persons (
    entry_id INTEGER NOT NULL REFERENCES entries(id) ON DELETE CASCADE,
    person_id INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    PRIMARY KEY (entry_id, person_id)
);

-- ═══════════════════════════════════════════════════════════════
-- NORMALIZED PLACES (many-to-many with entries)
-- ═══════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS places (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS entry_places (
    entry_id INTEGER NOT NULL REFERENCES entries(id) ON DELETE CASCADE,
    place_id INTEGER NOT NULL REFERENCES places(id) ON DELETE CASCADE,
    PRIMARY KEY (entry_id, place_id)
);

-- ═══════════════════════════════════════════════════════════════
-- DOCUMENT LINKS (one-to-many: entry has multiple image links)
-- ═══════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id INTEGER NOT NULL REFERENCES entries(id) ON DELETE CASCADE,
    url TEXT NOT NULL,
    link_text TEXT
);

-- ═══════════════════════════════════════════════════════════════
-- SCRAPE PROGRESS TRACKING (for resumability)
-- Year-level jobs track overall year progress
-- ═══════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS scrape_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    year INTEGER NOT NULL,
    county TEXT,
    status TEXT DEFAULT 'pending',
    results_count INTEGER DEFAULT 0,
    new_entries_count INTEGER DEFAULT 0,
    duplicate_count INTEGER DEFAULT 0,
    error_message TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    UNIQUE(year, county)
);

-- ═══════════════════════════════════════════════════════════════
-- PREFIX-LEVEL PROGRESS TRACKING (for alphabet prefix queries)
-- Tracks each year + surname prefix combination (a*, b*, c*, etc.)
-- ═══════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS scrape_prefix_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    year INTEGER NOT NULL,
    prefix TEXT NOT NULL,
    status TEXT DEFAULT 'pending',
    results_count INTEGER DEFAULT 0,
    new_entries_count INTEGER DEFAULT 0,
    duplicate_count INTEGER DEFAULT 0,
    error_message TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    UNIQUE(year, prefix)
);

-- ═══════════════════════════════════════════════════════════════
-- NORMALIZED SURNAMES (extracted from persons.name - last word)
-- ═══════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS surnames (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    surname TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS person_surnames (
    person_id INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    surname_id INTEGER NOT NULL REFERENCES surnames(id) ON DELETE CASCADE,
    PRIMARY KEY (person_id, surname_id)
);

-- ═══════════════════════════════════════════════════════════════
-- INDEXES FOR PERFORMANCE
-- ═══════════════════════════════════════════════════════════════
CREATE INDEX IF NOT EXISTS idx_entries_year ON entries(year);
CREATE INDEX IF NOT EXISTS idx_entries_county ON entries(county);
CREATE INDEX IF NOT EXISTS idx_entries_roll_ref ON entries(roll_reference);
CREATE INDEX IF NOT EXISTS idx_persons_name ON persons(name);
CREATE INDEX IF NOT EXISTS idx_places_name ON places(name);
CREATE INDEX IF NOT EXISTS idx_surnames_surname ON surnames(surname);
CREATE INDEX IF NOT EXISTS idx_scrape_jobs_status ON scrape_jobs(status);
CREATE INDEX IF NOT EXISTS idx_scrape_jobs_year ON scrape_jobs(year);
CREATE INDEX IF NOT EXISTS idx_scrape_prefix_jobs_year ON scrape_prefix_jobs(year);
CREATE INDEX IF NOT EXISTS idx_scrape_prefix_jobs_status ON scrape_prefix_jobs(status);
"""

# Alphabet prefixes for systematic querying
SURNAME_PREFIXES = list("abcdefghijklmnopqrstuvwxyz")


class CP40Database:
    """SQLite database handler for CP40 records"""
    
    def __init__(self, db_path: str = "cp40_records.db"):
        """Initialize database connection and create schema if needed"""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._create_schema()
    
    def _create_schema(self):
        """Create database schema if it doesn't exist"""
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()
    
    def close(self):
        """Close database connection"""
        self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    @staticmethod
    def compute_hash(text: str) -> str:
        """Compute SHA256 hash of text for deduplication"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def entry_exists(self, raw_text: str) -> bool:
        """Check if an entry with this raw_text already exists"""
        text_hash = self.compute_hash(raw_text)
        cursor = self.conn.execute(
            "SELECT 1 FROM entries WHERE raw_text_hash = ?",
            (text_hash,)
        )
        return cursor.fetchone() is not None
    
    def insert_entry(self, entry: Dict) -> Tuple[Optional[int], bool]:
        """
        Insert a single entry into the database
        
        Returns:
            Tuple of (entry_id, is_new) where is_new is False if duplicate
        """
        raw_text = entry.get('raw_text', '')
        if not raw_text:
            return None, False
        
        text_hash = self.compute_hash(raw_text)
        
        # Check for duplicate
        cursor = self.conn.execute(
            "SELECT id FROM entries WHERE raw_text_hash = ?",
            (text_hash,)
        )
        existing = cursor.fetchone()
        if existing:
            return existing['id'], False
        
        # Insert new entry
        cursor = self.conn.execute(
            """INSERT INTO entries (roll_reference, raw_text, raw_text_hash, county, year)
               VALUES (?, ?, ?, ?, ?)""",
            (
                entry.get('roll_reference', ''),
                raw_text,
                text_hash,
                entry.get('county'),
                int(entry['year']) if entry.get('year') else None
            )
        )
        entry_id = cursor.lastrowid
        
        # Insert persons
        for person_name in entry.get('persons', []):
            if person_name:
                person_id = self._get_or_create_person(person_name)
                self.conn.execute(
                    "INSERT OR IGNORE INTO entry_persons (entry_id, person_id) VALUES (?, ?)",
                    (entry_id, person_id)
                )
        
        # Insert places
        for place_name in entry.get('places', []):
            if place_name:
                place_id = self._get_or_create_place(place_name)
                self.conn.execute(
                    "INSERT OR IGNORE INTO entry_places (entry_id, place_id) VALUES (?, ?)",
                    (entry_id, place_id)
                )
        
        # Insert links
        for link in entry.get('links', []):
            self.conn.execute(
                "INSERT INTO links (entry_id, url, link_text) VALUES (?, ?, ?)",
                (entry_id, link.get('url', ''), link.get('text', ''))
            )
        
        return entry_id, True
    
    def _get_or_create_person(self, name: str) -> int:
        """Get existing person ID or create new one, also links surname"""
        cursor = self.conn.execute(
            "SELECT id FROM persons WHERE name = ?", (name,)
        )
        row = cursor.fetchone()
        if row:
            person_id = row['id']
        else:
            cursor = self.conn.execute(
                "INSERT INTO persons (name) VALUES (?)", (name,)
            )
            person_id = cursor.lastrowid
        
        # Extract and link surname
        surname = self.extract_surname(name)
        if surname:
            surname_id = self._get_or_create_surname(surname)
            self._link_person_surname(person_id, surname_id)
        
        return person_id
    
    def _get_or_create_place(self, name: str) -> int:
        """Get existing place ID or create new one"""
        cursor = self.conn.execute(
            "SELECT id FROM places WHERE name = ?", (name,)
        )
        row = cursor.fetchone()
        if row:
            return row['id']
        
        cursor = self.conn.execute(
            "INSERT INTO places (name) VALUES (?)", (name,)
        )
        return cursor.lastrowid
    
    @staticmethod
    def extract_surname(full_name: str) -> Optional[str]:
        """
        Extract surname from a full name (last word).
        
        Examples:
            "John Smith" -> "Smith"
            "William de la Pole" -> "Pole"
            "Robert" -> "Robert" (single names treated as surname)
        
        Returns:
            The surname (last word) or None if name is empty/whitespace
        """
        if not full_name or not full_name.strip():
            return None
        
        parts = full_name.strip().split()
        if not parts:
            return None
        
        # Return the last word as the surname
        return parts[-1]
    
    def _get_or_create_surname(self, surname: str) -> int:
        """Get existing surname ID or create new one"""
        cursor = self.conn.execute(
            "SELECT id FROM surnames WHERE surname = ?", (surname,)
        )
        row = cursor.fetchone()
        if row:
            return row['id']
        
        cursor = self.conn.execute(
            "INSERT INTO surnames (surname) VALUES (?)", (surname,)
        )
        return cursor.lastrowid
    
    def _link_person_surname(self, person_id: int, surname_id: int):
        """Link a person to their surname in the junction table"""
        self.conn.execute(
            "INSERT OR IGNORE INTO person_surnames (person_id, surname_id) VALUES (?, ?)",
            (person_id, surname_id)
        )
    
    def populate_surnames_from_persons(self, batch_size: int = 1000) -> Tuple[int, int]:
        """
        Populate the surnames table from existing persons data.
        
        Extracts the surname (last word) from each person's name
        and creates the appropriate links.
        
        Args:
            batch_size: Number of records to process per batch
        
        Returns:
            Tuple of (persons_processed, surnames_created)
        """
        # Get all persons that don't yet have a surname link
        cursor = self.conn.execute("""
            SELECT p.id, p.name 
            FROM persons p
            LEFT JOIN person_surnames ps ON p.id = ps.person_id
            WHERE ps.person_id IS NULL
        """)
        
        persons_processed = 0
        surnames_created_before = self.conn.execute(
            "SELECT COUNT(*) as count FROM surnames"
        ).fetchone()['count']
        
        batch = []
        for row in cursor:
            person_id = row['id']
            name = row['name']
            
            surname = self.extract_surname(name)
            if surname:
                batch.append((person_id, surname))
            
            if len(batch) >= batch_size:
                self._process_surname_batch(batch)
                persons_processed += len(batch)
                batch = []
        
        # Process remaining batch
        if batch:
            self._process_surname_batch(batch)
            persons_processed += len(batch)
        
        self.conn.commit()
        
        surnames_created_after = self.conn.execute(
            "SELECT COUNT(*) as count FROM surnames"
        ).fetchone()['count']
        
        return persons_processed, surnames_created_after - surnames_created_before
    
    def _process_surname_batch(self, batch: List[Tuple[int, str]]):
        """Process a batch of (person_id, surname) pairs"""
        for person_id, surname in batch:
            surname_id = self._get_or_create_surname(surname)
            self._link_person_surname(person_id, surname_id)
    
    def insert_entries(self, entries: List[Dict]) -> Tuple[int, int]:
        """
        Insert multiple entries, handling duplicates
        
        Returns:
            Tuple of (new_count, duplicate_count)
        """
        new_count = 0
        duplicate_count = 0
        
        for entry in entries:
            _, is_new = self.insert_entry(entry)
            if is_new:
                new_count += 1
            else:
                duplicate_count += 1
        
        self.conn.commit()
        return new_count, duplicate_count
    
    # ═══════════════════════════════════════════════════════════════
    # SCRAPE JOB TRACKING
    # ═══════════════════════════════════════════════════════════════
    
    def create_scrape_job(self, year: int, county: Optional[str] = None) -> int:
        """Create a new scrape job entry"""
        cursor = self.conn.execute(
            """INSERT OR IGNORE INTO scrape_jobs (year, county, status)
               VALUES (?, ?, 'pending')""",
            (year, county)
        )
        self.conn.commit()
        
        # Get the job ID (whether just inserted or already existed)
        cursor = self.conn.execute(
            "SELECT id FROM scrape_jobs WHERE year = ? AND county IS ?",
            (year, county)
        )
        return cursor.fetchone()['id']
    
    def get_job_status(self, year: int, county: Optional[str] = None) -> Optional[str]:
        """Get status of a scrape job"""
        cursor = self.conn.execute(
            "SELECT status FROM scrape_jobs WHERE year = ? AND county IS ?",
            (year, county)
        )
        row = cursor.fetchone()
        return row['status'] if row else None
    
    def start_scrape_job(self, year: int, county: Optional[str] = None):
        """Mark a scrape job as in progress"""
        self.conn.execute(
            """UPDATE scrape_jobs 
               SET status = 'in_progress', started_at = ?
               WHERE year = ? AND county IS ?""",
            (datetime.now().isoformat(), year, county)
        )
        self.conn.commit()
    
    def complete_scrape_job(
        self, 
        year: int, 
        county: Optional[str] = None,
        results_count: int = 0,
        new_entries: int = 0,
        duplicates: int = 0
    ):
        """Mark a scrape job as completed"""
        self.conn.execute(
            """UPDATE scrape_jobs 
               SET status = 'completed', 
                   completed_at = ?,
                   results_count = ?,
                   new_entries_count = ?,
                   duplicate_count = ?
               WHERE year = ? AND county IS ?""",
            (datetime.now().isoformat(), results_count, new_entries, duplicates, year, county)
        )
        self.conn.commit()
    
    def fail_scrape_job(self, year: int, county: Optional[str] = None, error: str = ""):
        """Mark a scrape job as failed"""
        self.conn.execute(
            """UPDATE scrape_jobs 
               SET status = 'failed', error_message = ?, completed_at = ?
               WHERE year = ? AND county IS ?""",
            (error, datetime.now().isoformat(), year, county)
        )
        self.conn.commit()
    
    def get_pending_years(self) -> List[int]:
        """Get list of years that haven't been scraped yet"""
        cursor = self.conn.execute(
            """SELECT DISTINCT year FROM scrape_jobs 
               WHERE status IN ('pending', 'failed') AND county IS NULL
               ORDER BY year"""
        )
        return [row['year'] for row in cursor.fetchall()]
    
    def get_incomplete_jobs(self) -> List[Dict]:
        """Get all incomplete jobs (pending or failed)"""
        cursor = self.conn.execute(
            """SELECT * FROM scrape_jobs 
               WHERE status IN ('pending', 'failed')
               ORDER BY year, county"""
        )
        return [dict(row) for row in cursor.fetchall()]
    
    def initialize_year_jobs(self, start_year: int = YEAR_START, end_year: int = YEAR_END):
        """Create pending jobs for all years in range"""
        for year in range(start_year, end_year + 1):
            self.create_scrape_job(year, None)
        self.conn.commit()
    
    # ═══════════════════════════════════════════════════════════════
    # PREFIX-LEVEL JOB TRACKING (for alphabet prefix queries)
    # ═══════════════════════════════════════════════════════════════
    
    def create_prefix_job(self, year: int, prefix: str) -> int:
        """Create a new prefix-level scrape job entry"""
        cursor = self.conn.execute(
            """INSERT OR IGNORE INTO scrape_prefix_jobs (year, prefix, status)
               VALUES (?, ?, 'pending')""",
            (year, prefix)
        )
        self.conn.commit()
        
        cursor = self.conn.execute(
            "SELECT id FROM scrape_prefix_jobs WHERE year = ? AND prefix = ?",
            (year, prefix)
        )
        return cursor.fetchone()['id']
    
    def get_prefix_job_status(self, year: int, prefix: str) -> Optional[str]:
        """Get status of a prefix-level scrape job"""
        cursor = self.conn.execute(
            "SELECT status FROM scrape_prefix_jobs WHERE year = ? AND prefix = ?",
            (year, prefix)
        )
        row = cursor.fetchone()
        return row['status'] if row else None
    
    def start_prefix_job(self, year: int, prefix: str):
        """Mark a prefix job as in progress"""
        self.conn.execute(
            """UPDATE scrape_prefix_jobs 
               SET status = 'in_progress', started_at = ?
               WHERE year = ? AND prefix = ?""",
            (datetime.now().isoformat(), year, prefix)
        )
        self.conn.commit()
    
    def complete_prefix_job(
        self, 
        year: int, 
        prefix: str,
        results_count: int = 0,
        new_entries: int = 0,
        duplicates: int = 0
    ):
        """Mark a prefix job as completed"""
        self.conn.execute(
            """UPDATE scrape_prefix_jobs 
               SET status = 'completed', 
                   completed_at = ?,
                   results_count = ?,
                   new_entries_count = ?,
                   duplicate_count = ?
               WHERE year = ? AND prefix = ?""",
            (datetime.now().isoformat(), results_count, new_entries, duplicates, year, prefix)
        )
        self.conn.commit()
    
    def fail_prefix_job(self, year: int, prefix: str, error: str = ""):
        """Mark a prefix job as failed"""
        self.conn.execute(
            """UPDATE scrape_prefix_jobs 
               SET status = 'failed', error_message = ?, completed_at = ?
               WHERE year = ? AND prefix = ?""",
            (error, datetime.now().isoformat(), year, prefix)
        )
        self.conn.commit()
    
    def initialize_prefix_jobs_for_year(self, year: int):
        """Create pending prefix jobs for all letters for a given year"""
        for prefix in SURNAME_PREFIXES:
            self.create_prefix_job(year, prefix)
        self.conn.commit()
    
    def get_pending_prefixes_for_year(self, year: int) -> List[str]:
        """Get list of prefixes that haven't been completed for a year"""
        cursor = self.conn.execute(
            """SELECT prefix FROM scrape_prefix_jobs 
               WHERE year = ? AND status IN ('pending', 'failed')
               ORDER BY prefix""",
            (year,)
        )
        return [row['prefix'] for row in cursor.fetchall()]
    
    def get_year_prefix_progress(self, year: int) -> Dict:
        """Get progress summary for a year's prefix jobs"""
        cursor = self.conn.execute(
            """SELECT 
                 COUNT(*) as total,
                 SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                 SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending,
                 SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
                 SUM(results_count) as total_results,
                 SUM(new_entries_count) as total_new
               FROM scrape_prefix_jobs 
               WHERE year = ?""",
            (year,)
        )
        row = cursor.fetchone()
        return dict(row) if row else {}
    
    # ═══════════════════════════════════════════════════════════════
    # STATISTICS AND QUERIES
    # ═══════════════════════════════════════════════════════════════
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        stats = {}
        
        # Total entries
        cursor = self.conn.execute("SELECT COUNT(*) as count FROM entries")
        stats['total_entries'] = cursor.fetchone()['count']
        
        # Entries by year range
        cursor = self.conn.execute(
            "SELECT MIN(year) as min_year, MAX(year) as max_year FROM entries WHERE year IS NOT NULL"
        )
        row = cursor.fetchone()
        stats['min_year'] = row['min_year']
        stats['max_year'] = row['max_year']
        
        # Total persons
        cursor = self.conn.execute("SELECT COUNT(*) as count FROM persons")
        stats['total_persons'] = cursor.fetchone()['count']
        
        # Total places
        cursor = self.conn.execute("SELECT COUNT(*) as count FROM places")
        stats['total_places'] = cursor.fetchone()['count']
        
        # Total surnames
        cursor = self.conn.execute("SELECT COUNT(*) as count FROM surnames")
        stats['total_surnames'] = cursor.fetchone()['count']
        
        # Scrape progress (year-level)
        cursor = self.conn.execute(
            """SELECT status, COUNT(*) as count 
               FROM scrape_jobs 
               GROUP BY status"""
        )
        stats['jobs_by_status'] = {row['status']: row['count'] for row in cursor.fetchall()}
        
        # Scrape progress (prefix-level)
        cursor = self.conn.execute(
            """SELECT status, COUNT(*) as count 
               FROM scrape_prefix_jobs 
               GROUP BY status"""
        )
        stats['prefix_jobs_by_status'] = {row['status']: row['count'] for row in cursor.fetchall()}
        
        # Entries by county
        cursor = self.conn.execute(
            """SELECT county, COUNT(*) as count 
               FROM entries 
               GROUP BY county 
               ORDER BY count DESC 
               LIMIT 10"""
        )
        stats['top_counties'] = [(row['county'], row['count']) for row in cursor.fetchall()]
        
        return stats
    
    def search_persons(self, name_pattern: str, limit: int = 100) -> List[Dict]:
        """Search for persons by name pattern (uses LIKE)"""
        cursor = self.conn.execute(
            """SELECT p.id, p.name, COUNT(ep.entry_id) as entry_count
               FROM persons p
               LEFT JOIN entry_persons ep ON p.id = ep.person_id
               WHERE p.name LIKE ?
               GROUP BY p.id
               ORDER BY entry_count DESC
               LIMIT ?""",
            (f"%{name_pattern}%", limit)
        )
        return [dict(row) for row in cursor.fetchall()]
    
    def search_places(self, name_pattern: str, limit: int = 100) -> List[Dict]:
        """Search for places by name pattern (uses LIKE)"""
        cursor = self.conn.execute(
            """SELECT p.id, p.name, COUNT(ep.entry_id) as entry_count
               FROM places p
               LEFT JOIN entry_places ep ON p.id = ep.place_id
               WHERE p.name LIKE ?
               GROUP BY p.id
               ORDER BY entry_count DESC
               LIMIT ?""",
            (f"%{name_pattern}%", limit)
        )
        return [dict(row) for row in cursor.fetchall()]
    
    def search_surnames(self, surname_pattern: str, limit: int = 100) -> List[Dict]:
        """Search for surnames by pattern (uses LIKE)"""
        cursor = self.conn.execute(
            """SELECT s.id, s.surname, COUNT(ps.person_id) as person_count
               FROM surnames s
               LEFT JOIN person_surnames ps ON s.id = ps.surname_id
               WHERE s.surname LIKE ?
               GROUP BY s.id
               ORDER BY person_count DESC
               LIMIT ?""",
            (f"%{surname_pattern}%", limit)
        )
        return [dict(row) for row in cursor.fetchall()]
    
    def get_persons_by_surname(self, surname: str, limit: int = 100) -> List[Dict]:
        """Get all persons with a specific surname"""
        cursor = self.conn.execute(
            """SELECT p.id, p.name
               FROM persons p
               JOIN person_surnames ps ON p.id = ps.person_id
               JOIN surnames s ON ps.surname_id = s.id
               WHERE s.surname = ?
               LIMIT ?""",
            (surname, limit)
        )
        return [dict(row) for row in cursor.fetchall()]
    
    def get_entries_for_person(self, person_name: str, limit: int = 100) -> List[Dict]:
        """Get all entries mentioning a specific person"""
        cursor = self.conn.execute(
            """SELECT e.*
               FROM entries e
               JOIN entry_persons ep ON e.id = ep.entry_id
               JOIN persons p ON ep.person_id = p.id
               WHERE p.name = ?
               ORDER BY e.year
               LIMIT ?""",
            (person_name, limit)
        )
        return [dict(row) for row in cursor.fetchall()]
    
    def get_entries_by_year(self, year: int, limit: int = 1000) -> List[Dict]:
        """Get all entries for a specific year"""
        cursor = self.conn.execute(
            "SELECT * FROM entries WHERE year = ? LIMIT ?",
            (year, limit)
        )
        return [dict(row) for row in cursor.fetchall()]


def main():
    """Test database functionality"""
    import json
    
    db_path = "cp40_records.db"
    print(f"Initializing database: {db_path}")
    
    with CP40Database(db_path) as db:
        # Initialize jobs for all years
        print(f"Creating scrape jobs for years {YEAR_START}-{YEAR_END}...")
        db.initialize_year_jobs()
        
        # Show stats
        stats = db.get_stats()
        print("\nDatabase Statistics:")
        print(json.dumps(stats, indent=2, default=str))
        
        # Show pending jobs
        pending = db.get_pending_years()
        print(f"\nPending years to scrape: {len(pending)}")
        if pending:
            print(f"  First 10: {pending[:10]}")


if __name__ == '__main__':
    main()

