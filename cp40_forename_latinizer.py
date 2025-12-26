#!/usr/bin/env python3
"""
CP40 Forename Latinizer
Extracts forenames from the CP40 database and generates Latin declined/abbreviated
forms using Gemini 3 Flash, then expands with heuristic-based orthographic variants.

Variant Generation Strategy:
- LLM: Used ONLY for English→Latin mapping (John→Johannes) - requires linguistic knowledge
- Heuristics: Used for ALL orthographic variants - systematic and reliable:
  - I/J interchange (medieval Latin didn't distinguish)
  - U/V interchange (especially initial position)  
  - Multiple abbreviation patterns
  - Case normalization

Usage:
    # Extract forenames and populate Latin forms
    python cp40_forename_latinizer.py
    
    # Show statistics only
    python cp40_forename_latinizer.py --stats
    
    # Process specific forenames
    python cp40_forename_latinizer.py --forenames "John,William,Thomas"
    
    # Resume processing (skip already completed)
    python cp40_forename_latinizer.py --resume
    
    # Look up all variants for a name
    python cp40_forename_latinizer.py --lookup "Johannes"
"""

import argparse
import os
import sqlite3
import json
import re
import time
import random
from typing import List, Dict, Optional, Set, Tuple
from datetime import datetime
from itertools import product
from google import genai
from google.genai import types


def get_free_api_key() -> str:
    """
    Get a free API key for Gemini API calls.
    Reads from environment variable GEMINI_FREE_API_KEYS (comma-separated list).
    If not set, falls back to GEMINI_API_KEY.
    Randomly selects from available keys to distribute load.
    
    Returns:
        str: A free API key for Gemini API
    
    Raises:
        RuntimeError: If no API keys are configured
    """
    import os
    
    # Try to get free API keys from environment (comma-separated)
    free_keys_str = os.environ.get('GEMINI_FREE_API_KEYS', '')
    if free_keys_str:
        free_keys = [key.strip() for key in free_keys_str.split(',') if key.strip()]
        if free_keys:
            return random.choice(free_keys)
    
    # Fall back to main API key
    main_key = os.environ.get('GEMINI_API_KEY')
    if main_key:
        return main_key
    
    raise RuntimeError(
        "No API keys configured. Set GEMINI_FREE_API_KEYS (comma-separated) "
        "or GEMINI_API_KEY environment variable"
    )


# Latin declension cases used in medieval legal documents
LATIN_CASES = [
    'nominative',   # Subject case (Johannes)
    'genitive',     # Possessive "of X" (Johannis)
    'dative',       # Indirect object "to/for X" (Johanni)
    'accusative',   # Direct object (Johannem)
    'ablative',     # "by/with/from X" (Johanne)
]

# Base schema for forename Latin forms (compatible with old and new databases)
# Note: New columns (normalized_form, variant_type) are added via migrations
FORENAME_SCHEMA_SQL = """
-- ═══════════════════════════════════════════════════════════════
-- FORENAMES TABLE (unique English/anglicized forenames)
-- ═══════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS forenames (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    english_name TEXT NOT NULL UNIQUE,
    frequency INTEGER DEFAULT 0,
    gender TEXT,  -- 'm', 'f', or NULL if unknown
    processed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ═══════════════════════════════════════════════════════════════
-- LATIN FORMS TABLE (declined, abbreviated Latin versions)
-- Each forename can have multiple Latin forms per declension
-- Note: normalized_form and variant_type columns added via migration
-- ═══════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS forename_latin_forms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    forename_id INTEGER NOT NULL REFERENCES forenames(id) ON DELETE CASCADE,
    case_name TEXT NOT NULL,  -- nominative, genitive, dative, accusative, ablative
    latin_full TEXT NOT NULL,  -- Full Latin form (e.g., "Johannes" or "Iohannes")
    latin_abbreviated TEXT NOT NULL,  -- Abbreviated form with apostrophe (e.g., "Joh'es")
    is_primary BOOLEAN DEFAULT 0,  -- Primary/canonical form from LLM
    notes TEXT,  -- Any special notes about this form
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(forename_id, case_name, latin_abbreviated)
);

-- ═══════════════════════════════════════════════════════════════
-- PROCESSING STATUS TABLE (track API call progress)
-- ═══════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS forename_processing_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    forename_id INTEGER NOT NULL REFERENCES forenames(id) ON DELETE CASCADE,
    status TEXT DEFAULT 'pending',  -- pending, in_progress, completed, failed
    error_message TEXT,
    api_response TEXT,  -- Store raw API response for debugging
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    UNIQUE(forename_id)
);

-- ═══════════════════════════════════════════════════════════════
-- INDEXES (only on columns that exist in base schema)
-- ═══════════════════════════════════════════════════════════════
CREATE INDEX IF NOT EXISTS idx_forenames_english ON forenames(english_name);
CREATE INDEX IF NOT EXISTS idx_forenames_frequency ON forenames(frequency DESC);
CREATE INDEX IF NOT EXISTS idx_latin_forms_forename ON forename_latin_forms(forename_id);
CREATE INDEX IF NOT EXISTS idx_latin_forms_case ON forename_latin_forms(case_name);
CREATE INDEX IF NOT EXISTS idx_latin_forms_abbreviated ON forename_latin_forms(latin_abbreviated);
CREATE INDEX IF NOT EXISTS idx_processing_status ON forename_processing_jobs(status);
"""


class LatinVariantGenerator:
    """
    Generates orthographic variants for Latin names using reliable heuristics.
    
    Medieval Latin orthography was not standardized. This class generates all
    plausible variants a scribe might have used, enabling robust matching.
    """
    
    @staticmethod
    def normalize(text: str) -> str:
        """
        Normalize a Latin form for consistent lookup.
        
        Normalization rules:
        - Lowercase
        - J → I (medieval Latin didn't have J)
        - V → U (in non-initial position, V was often used for U)
        - Remove apostrophes and other abbreviation markers
        - Strip whitespace
        
        Args:
            text: Latin text to normalize
            
        Returns:
            Normalized form for lookup
        """
        if not text:
            return ""
        
        normalized = text.lower()
        
        # J → I (standard medieval Latin normalization)
        normalized = normalized.replace('j', 'i')
        
        # V → U normalization (but keep initial V as it might be intentional)
        # Actually, for matching purposes, normalize all V→U
        normalized = normalized.replace('v', 'u')
        
        # Remove abbreviation markers
        normalized = normalized.replace("'", "")
        normalized = normalized.replace("'", "")  # curly apostrophe
        normalized = normalized.replace("`", "")
        
        # Remove common superscript/abbreviation markers
        normalized = re.sub(r'[̃˜~]', '', normalized)  # tildes
        
        return normalized.strip()
    
    @staticmethod
    def generate_ij_variants(text: str) -> Set[str]:
        """
        Generate I/J interchange variants.
        
        In medieval Latin, I and J were not distinct letters. J developed
        as a variant of I, particularly at the start of words before vowels.
        
        Examples:
        - Johannes ↔ Iohannes
        - Jacobus ↔ Iacobus
        - major ↔ maior
        
        Args:
            text: Latin text
            
        Returns:
            Set of all I/J variants including the original
        """
        if not text:
            return {text} if text else set()
        
        variants = set()
        
        # Find all positions with I or J
        positions = []
        for i, char in enumerate(text):
            if char in 'IiJj':
                positions.append(i)
        
        if not positions:
            return {text}
        
        # Generate all combinations of I/J at each position
        for combo in product(['I', 'J'], repeat=len(positions)):
            variant = list(text)
            for pos_idx, char in zip(positions, combo):
                # Preserve original case
                if text[pos_idx].isupper():
                    variant[pos_idx] = char.upper()
                else:
                    variant[pos_idx] = char.lower()
            variants.add(''.join(variant))
        
        return variants
    
    @staticmethod
    def generate_uv_variants(text: str) -> Set[str]:
        """
        Generate U/V interchange variants.
        
        In classical and medieval Latin:
        - V was used for both U and V sounds
        - Initial U was often written as V
        - Medial V might be written as U
        
        Common patterns:
        - Initial position: prefer V (Villelmus)
        - Medial position: could be either
        - After certain consonants: U more common
        
        Args:
            text: Latin text
            
        Returns:
            Set of all U/V variants including the original
        """
        if not text:
            return {text} if text else set()
        
        variants = set()
        
        # Find all positions with U or V
        positions = []
        for i, char in enumerate(text):
            if char in 'UuVv':
                positions.append(i)
        
        if not positions:
            return {text}
        
        # Generate all combinations of U/V at each position
        for combo in product(['U', 'V'], repeat=len(positions)):
            variant = list(text)
            for pos_idx, char in zip(positions, combo):
                # Preserve original case
                if text[pos_idx].isupper():
                    variant[pos_idx] = char.upper()
                else:
                    variant[pos_idx] = char.lower()
            variants.add(''.join(variant))
        
        return variants
    
    @staticmethod
    def generate_double_letter_variants(text: str) -> Set[str]:
        """
        Generate double/single letter variants for common patterns.
        
        Medieval scribes were inconsistent with double letters:
        - Willelmus ↔ Wilhelmus ↔ Willelmus
        - Iohannes ↔ Ioannes (double n)
        - Philippus ↔ Philipus
        
        Args:
            text: Latin text
            
        Returns:
            Set of variants with double letter alternations
        """
        if not text:
            return {text} if text else set()
        
        variants = {text}
        
        # Common double letter positions in Latin names
        double_patterns = [
            (r'([aeiou])nn([aeiou])', r'\1n\2'),   # nn → n between vowels
            (r'([aeiou])n([aeiou])', r'\1nn\2'),   # n → nn between vowels
            (r'll', 'l'),                          # ll → l
            (r'([^l])l([^l])', r'\1ll\2'),         # l → ll (not already double)
            (r'pp', 'p'),                          # pp → p
            (r'([^p])p([^p])', r'\1pp\2'),         # p → pp
            (r'tt', 't'),                          # tt → t
            (r'ss', 's'),                          # ss → s
        ]
        
        for pattern, replacement in double_patterns:
            new_variant = re.sub(pattern, replacement, text, count=1)
            if new_variant != text:
                variants.add(new_variant)
        
        return variants
    
    @staticmethod  
    def generate_abbreviation_variants(full_form: str, primary_abbrev: str) -> Set[str]:
        """
        Generate multiple plausible abbreviation variants.
        
        Medieval abbreviation patterns:
        1. Contraction with apostrophe: Joh'es (Johannes)
        2. Suspension (truncation): Joh' or Jo.
        3. Special marks: superscript letters, tildes
        4. First + last syllable: Joh...es
        5. Suspension + case ending: Rob't → Rob'tus (adding back the ending)
        
        We focus on apostrophe-based contractions since that's searchable.
        
        Args:
            full_form: Full Latin form (e.g., "Johannes")
            primary_abbrev: Primary abbreviation from LLM (e.g., "Joh'es")
            
        Returns:
            Set of plausible abbreviation variants
        """
        if not full_form:
            return {primary_abbrev} if primary_abbrev else set()
        
        variants = {primary_abbrev} if primary_abbrev else set()
        full_lower = full_form.lower()
        
        # Latin case endings (2nd declension masculine/neuter, 1st declension feminine, 3rd declension)
        case_endings_2m = ['us', 'i', 'o', 'um', 'o']  # nom, gen, dat, acc, abl
        case_endings_2n = ['um', 'i', 'o', 'um', 'o']
        case_endings_1f = ['a', 'ae', 'ae', 'am', 'a']
        case_endings_3 = ['', 'is', 'i', 'em', 'e']
        all_endings = set(case_endings_2m + case_endings_2n + case_endings_1f + case_endings_3)
        all_endings.update(['es', 'as', 'os', 'is', 'nis', 'tis', 'mis', 'lis', 'tus', 'tum', 'ti', 'to'])
        
        # CRITICAL: If primary_abbrev ends without a case ending (like Rob't),
        # generate variants by adding case endings back (Rob't → Rob'tus, Rob'ti, etc.)
        if primary_abbrev and "'" in primary_abbrev:
            # Check if abbreviation ends with consonant (suspension without ending)
            last_char = primary_abbrev[-1].lower()
            if last_char not in 'aeioum':  # Ends with consonant - likely suspended
                # Try adding case endings
                for ending in ['us', 'um', 'i', 'o', 'e', 'is', 'em', 'a', 'ae', 'am']:
                    variants.add(primary_abbrev + ending)
        
        # Common abbreviation patterns based on word structure
        # Pattern: first N letters + apostrophe + last M letters
        
        if len(full_form) >= 4:
            # Get the ending (typically last 2-3 chars for case endings)
            endings = []
            if full_lower.endswith(('us', 'um', 'is', 'em', 'am', 'ae', 'os')):
                endings.append(full_form[-2:])
            if full_lower.endswith(('es', 'as', 'os', 'is')):
                endings.append(full_form[-2:])
            if full_lower.endswith(('i', 'o', 'e', 'a', 'u')):
                endings.append(full_form[-1:])
            if full_lower.endswith(('nis', 'tis', 'mis', 'lis', 'tus', 'tum')):
                endings.append(full_form[-3:])
            
            # Common prefix lengths to try
            prefix_lengths = [2, 3, 4, 5]
            
            for prefix_len in prefix_lengths:
                if prefix_len < len(full_form):
                    prefix = full_form[:prefix_len]
                    for ending in endings:
                        # Make sure we're actually abbreviating
                        if len(prefix) + len(ending) + 1 < len(full_form):
                            abbrev = f"{prefix}'{ending}"
                            variants.add(abbrev)
            
            # Also try prefix + apostrophe + longer endings (for forms like Rob'tus)
            # This catches cases where the abbreviation preserves part of the stem
            for stem_len in range(3, min(len(full_form) - 2, 6)):
                stem = full_form[:stem_len]
                remaining = full_form[stem_len:]
                if len(remaining) >= 2:
                    # Add apostrophe after stem
                    abbrev = f"{stem}'{remaining}"
                    if abbrev != full_form and len(abbrev) < len(full_form):
                        variants.add(abbrev)
        
        # Single apostrophe at end (suspension)
        for i in range(2, min(6, len(full_form))):
            variants.add(full_form[:i] + "'")
        
        # CRITICAL: Also add the full unabbreviated form!
        # Medieval documents often have both abbreviated AND full forms
        # e.g., "Hugoni" (full) and "Hug'i" (abbreviated) are both valid
        if full_form and len(full_form) >= 3:
            variants.add(full_form)
        
        # Add suspension variants that cut off just the case ending
        # e.g., Hugoni → Hugon' (cut the final 'i')
        # This is common when scribes abbreviated by dropping just the last letter(s)
        if len(full_form) >= 4:
            for cut_len in [1, 2]:
                if len(full_form) > cut_len + 2:
                    suspended = full_form[:-cut_len] + "'"
                    variants.add(suspended)
        
        return variants
    
    @classmethod
    def generate_all_variants(
        cls, 
        full_form: str, 
        abbreviated: str
    ) -> List[Tuple[str, str, str]]:
        """
        Generate all orthographic variants for a Latin form.
        
        Args:
            full_form: Full Latin form (e.g., "Johannes")
            abbreviated: Primary abbreviated form (e.g., "Joh'es")
            
        Returns:
            List of tuples: (full_variant, abbrev_variant, variant_type)
            where variant_type is one of:
            - 'primary': Original form from LLM
            - 'ij_variant': I/J interchange
            - 'uv_variant': U/V interchange
            - 'ij_uv_variant': Both I/J and U/V interchange
            - 'abbrev_variant': Alternative abbreviation pattern
        """
        results = []
        
        # Primary form
        results.append((full_form, abbreviated, 'primary'))
        
        # Generate I/J variants for full form
        ij_full_variants = cls.generate_ij_variants(full_form)
        
        # Generate U/V variants for each I/J variant
        all_full_variants = set()
        for ij_var in ij_full_variants:
            uv_vars = cls.generate_uv_variants(ij_var)
            all_full_variants.update(uv_vars)
        
        # For each full form variant, generate abbreviation variants
        for full_var in all_full_variants:
            # Determine variant type based on what changed
            is_ij = (full_var.lower().replace('j', 'i') != 
                     full_form.lower().replace('j', 'i').replace('i', 'i'))
            has_ij_diff = any(
                (c1.lower() in 'ij' and c2.lower() in 'ij' and c1.lower() != c2.lower())
                for c1, c2 in zip(full_form, full_var) if len(full_form) == len(full_var)
            )
            has_uv_diff = any(
                (c1.lower() in 'uv' and c2.lower() in 'uv' and c1.lower() != c2.lower())
                for c1, c2 in zip(full_form, full_var) if len(full_form) == len(full_var)
            )
            
            if has_ij_diff and has_uv_diff:
                var_type = 'ij_uv_variant'
            elif has_ij_diff:
                var_type = 'ij_variant'
            elif has_uv_diff:
                var_type = 'uv_variant'
            elif full_var == full_form:
                var_type = 'primary'
            else:
                var_type = 'spelling_variant'
            
            # Skip the primary which we already added
            if full_var == full_form and var_type == 'primary':
                continue
            
            # Generate corresponding abbreviation variant
            # Apply same transformations to abbreviation
            abbrev_var = abbreviated
            if has_ij_diff:
                # Apply I/J transformation to abbreviation too
                ij_abbrevs = cls.generate_ij_variants(abbreviated)
                for ij_abbr in ij_abbrevs:
                    if ij_abbr != abbreviated:
                        results.append((full_var, ij_abbr, var_type))
                abbrev_var = list(ij_abbrevs)[0] if ij_abbrevs else abbreviated
            
            if has_uv_diff:
                # Apply U/V transformation to abbreviation too
                uv_abbrevs = cls.generate_uv_variants(abbrev_var)
                for uv_abbr in uv_abbrevs:
                    if uv_abbr != abbreviated:
                        results.append((full_var, uv_abbr, var_type))
            
            # Add the main variant
            if full_var != full_form:
                results.append((full_var, abbrev_var, var_type))
        
        # Generate additional abbreviation patterns for primary form
        abbrev_variants = cls.generate_abbreviation_variants(full_form, abbreviated)
        for abbrev_var in abbrev_variants:
            if abbrev_var != abbreviated:
                results.append((full_form, abbrev_var, 'abbrev_variant'))
        
        # Deduplicate while preserving order and preferring 'primary' type
        seen = set()
        deduped = []
        for item in results:
            key = (item[0], item[1])
            if key not in seen:
                seen.add(key)
                deduped.append(item)
        
        return deduped


class ForenameLatinizer:
    """
    Extracts forenames from CP40 database and generates Latin forms using Gemini,
    then expands with heuristic-based orthographic variants.
    """
    
    def __init__(
        self, 
        db_path: str = "cp40_records.db",
        delay: float = 0.5,
        verbose: bool = True
    ):
        """
        Initialize the latinizer.
        
        Args:
            db_path: Path to the CP40 SQLite database
            delay: Delay between API calls in seconds
            verbose: Print progress information
        """
        self.db_path = db_path
        self.delay = delay
        self.verbose = verbose
        
        # Initialize variant generator
        self.variant_gen = LatinVariantGenerator()
        
        # Connect to database
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        
        # Create schema and run migrations
        self._create_schema()
        self._run_migrations()
        
        # Initialize Gemini client
        self.client = self._init_gemini_client()
    
    def _init_gemini_client(self) -> genai.Client:
        """Initialize Gemini API client using paid API key from environment variable."""
        import os
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable must be set with a paid API key")
        return genai.Client(api_key=api_key)
    
    def _create_schema(self):
        """Create the forename schema if it doesn't exist."""
        self.conn.executescript(FORENAME_SCHEMA_SQL)
        self.conn.commit()
    
    def _run_migrations(self):
        """Run database migrations for new columns."""
        migrations_run = []
        
        # Check forename_latin_forms columns
        cursor = self.conn.execute("PRAGMA table_info(forename_latin_forms)")
        latin_forms_columns = {row['name'] for row in cursor.fetchall()}
        
        # Add normalized_form column if missing
        if 'normalized_form' not in latin_forms_columns:
            try:
                self.conn.execute(
                    "ALTER TABLE forename_latin_forms ADD COLUMN normalized_form TEXT"
                )
                migrations_run.append("Added normalized_form column")
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise
        
        # Add variant_type column if missing
        if 'variant_type' not in latin_forms_columns:
            try:
                self.conn.execute(
                    "ALTER TABLE forename_latin_forms ADD COLUMN variant_type TEXT DEFAULT 'primary'"
                )
                migrations_run.append("Added variant_type column")
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise
        
        # Check processing_jobs table
        cursor = self.conn.execute("PRAGMA table_info(forename_processing_jobs)")
        job_columns = {row['name'] for row in cursor.fetchall()}
        
        if 'variants_generated' not in job_columns:
            try:
                self.conn.execute(
                    "ALTER TABLE forename_processing_jobs ADD COLUMN variants_generated INTEGER DEFAULT 0"
                )
                migrations_run.append("Added variants_generated column")
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise
        
        self.conn.commit()
        
        # Now create indexes (after columns exist)
        # Re-check columns after potential additions
        cursor = self.conn.execute("PRAGMA table_info(forename_latin_forms)")
        latin_forms_columns = {row['name'] for row in cursor.fetchall()}
        
        if 'normalized_form' in latin_forms_columns:
            try:
                self.conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_latin_forms_normalized ON forename_latin_forms(normalized_form)"
                )
            except sqlite3.OperationalError:
                pass
        
        if 'variant_type' in latin_forms_columns:
            try:
                self.conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_latin_forms_variant_type ON forename_latin_forms(variant_type)"
                )
            except sqlite3.OperationalError:
                pass
        
        # Backfill normalized_form for existing records that don't have it
        if 'normalized_form' in latin_forms_columns:
            self.conn.execute("""
                UPDATE forename_latin_forms 
                SET normalized_form = LOWER(REPLACE(REPLACE(REPLACE(latin_abbreviated, 'j', 'i'), 'J', 'I'), '''', ''))
                WHERE normalized_form IS NULL OR normalized_form = ''
            """)
        
        # Backfill variant_type for existing records
        if 'variant_type' in latin_forms_columns:
            self.conn.execute("""
                UPDATE forename_latin_forms 
                SET variant_type = 'primary'
                WHERE variant_type IS NULL
            """)
        
        self.conn.commit()
        
        if migrations_run:
            self.log(f"Migrations: {', '.join(migrations_run)}")
    
    def close(self):
        """Close database connection."""
        self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def log(self, message: str):
        """Print message if verbose mode is on."""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {message}")
    
    def extract_forenames_from_persons(self) -> int:
        """
        Extract unique forenames from the persons table and populate
        the forenames table.
        
        Returns:
            Number of new forenames added
        """
        self.log("Extracting forenames from persons table...")
        
        # Get all unique person names
        cursor = self.conn.execute(
            "SELECT name, COUNT(*) as freq FROM persons GROUP BY name"
        )
        
        forename_counts = {}
        
        for row in cursor.fetchall():
            name = row['name']
            freq = row['freq']
            
            # Extract forename (first word, handling common patterns)
            forename = self._extract_forename(name)
            
            if forename:
                # Normalize to title case
                forename = forename.strip().title()
                
                # Skip if too short or looks invalid
                if len(forename) < 2:
                    continue
                if not forename[0].isalpha():
                    continue
                    
                forename_counts[forename] = forename_counts.get(forename, 0) + freq
        
        # Insert into forenames table
        new_count = 0
        for forename, frequency in forename_counts.items():
            try:
                self.conn.execute(
                    """INSERT INTO forenames (english_name, frequency) 
                       VALUES (?, ?)
                       ON CONFLICT(english_name) DO UPDATE SET 
                       frequency = frequency + excluded.frequency""",
                    (forename, frequency)
                )
                new_count += 1
            except sqlite3.IntegrityError:
                pass
        
        self.conn.commit()
        
        # Count total unique forenames
        cursor = self.conn.execute("SELECT COUNT(*) FROM forenames")
        total = cursor.fetchone()[0]
        
        self.log(f"Extracted {total} unique forenames ({new_count} new)")
        return new_count
    
    def _extract_forename(self, full_name: str) -> Optional[str]:
        """
        Extract the forename from a full name string.
        
        Handles patterns like:
        - "John Smith" -> "John"
        - "John de la Pole" -> "John"
        - "John (alias Jack) Smith" -> "John"
        - "Sir John Smith" -> "John"
        """
        if not full_name:
            return None
        
        name = full_name.strip()
        
        # Remove common titles/prefixes
        titles = ['sir', 'master', 'dame', 'lady', 'lord', 'brother', 'sister', 'fr.', 'fr']
        lower_name = name.lower()
        for title in titles:
            if lower_name.startswith(title + ' '):
                name = name[len(title) + 1:].strip()
                break
        
        # Remove parenthetical notes
        name = re.sub(r'\([^)]*\)', '', name).strip()
        
        # Split and get first word
        parts = name.split()
        if not parts:
            return None
        
        forename = parts[0]
        
        # Clean up any trailing punctuation
        forename = re.sub(r'[,;:]$', '', forename)
        
        return forename if forename else None
    
    def initialize_processing_jobs(self):
        """Create processing jobs for all forenames that don't have one."""
        self.conn.execute("""
            INSERT OR IGNORE INTO forename_processing_jobs (forename_id, status)
            SELECT id, 'pending' FROM forenames
            WHERE id NOT IN (SELECT forename_id FROM forename_processing_jobs)
        """)
        self.conn.commit()
    
    def get_pending_forenames(self, limit: Optional[int] = None) -> List[Dict]:
        """Get forenames that haven't been processed yet."""
        query = """
            SELECT f.id, f.english_name, f.frequency
            FROM forenames f
            JOIN forename_processing_jobs j ON f.id = j.forename_id
            WHERE j.status IN ('pending', 'failed')
            ORDER BY f.frequency DESC
        """
        if limit:
            query += f" LIMIT {limit}"
        
        cursor = self.conn.execute(query)
        return [dict(row) for row in cursor.fetchall()]
    
    def _build_prompt(self, forename: str) -> str:
        """Build the prompt for Gemini to generate Latin forms."""
        return f"""Medieval Latin forms for the English forename "{forename}".

Rules:
- Use straight apostrophe (') for abbreviated/dropped letters
- Use J (not I) for initial J sounds - we will generate I variants automatically
- Examples: Joh'es (Johannes), Will'm (Willelmum), Thom' (Thomas)

Return ONLY this compact JSON (one line per case, no extra whitespace):
{{"name":"{forename}","gender":"m or f","nom_full":"","nom_abbr":"","gen_full":"","gen_abbr":"","dat_full":"","dat_abbr":"","acc_full":"","acc_abbr":"","abl_full":"","abl_abbr":""}}

Fill in all fields. Example for John:
{{"name":"John","gender":"m","nom_full":"Johannes","nom_abbr":"Joh'es","gen_full":"Johannis","gen_abbr":"Joh'is","dat_full":"Johanni","dat_abbr":"Joh'i","acc_full":"Johannem","acc_abbr":"Joh'em","abl_full":"Johanne","abl_abbr":"Joh'e"}}"""
    
    def _call_gemini_api(self, forename: str, retry_count: int = 0) -> Optional[Dict]:
        """
        Call Gemini 3 Flash to get Latin forms for a forename.
        
        Returns:
            Parsed JSON response or None if failed
        """
        prompt = self._build_prompt(forename)
        
        # Don't use response_mime_type="application/json" as it can cause truncation issues
        # Free API call - no need to limit tokens
        config = types.GenerateContentConfig(
            temperature=0,  # Temperature 0 for deterministic output
            thinking_config=types.ThinkingConfig(thinking_level='low'),  # Low thinking mode
        )
        
        try:
            response = self.client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=[types.Part.from_text(text=prompt)],
                config=config,
            )
            
            if response and hasattr(response, 'text') and response.text:
                # Parse JSON response
                json_text = response.text.strip()
                
                # Remove markdown code fences if present
                if json_text.startswith('```'):
                    json_text = re.sub(r'^```(?:json)?\n?', '', json_text)
                    json_text = re.sub(r'\n?```$', '', json_text)
                
                # Replace fancy quotes with straight quotes
                json_text = json_text.replace('"', '"').replace('"', '"')
                json_text = json_text.replace(''', "'").replace(''', "'")
                
                # Handle multiple JSON objects (NDJSON/JSONL format)
                # Split by newlines and try to parse each line as a separate JSON object
                lines = json_text.split('\n')
                parsed_objects = []
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Try to extract JSON object from the line
                    json_match = re.search(r'\{[\s\S]*\}', line)
                    if json_match:
                        json_line = json_match.group(0)
                        try:
                            parsed_obj = json.loads(json_line)
                            parsed_objects.append(parsed_obj)
                        except json.JSONDecodeError:
                            # Skip invalid JSON lines
                            continue
                
                # If we found at least one valid JSON object, use the first one
                if parsed_objects:
                    # Prefer the first object (or could prefer a specific gender)
                    return parsed_objects[0]
                
                # Fallback: try to parse the entire text as a single JSON object
                try:
                    return json.loads(json_text)
                except json.JSONDecodeError as je:
                    # Log the raw response for debugging
                    if self.verbose:
                        self.log(f"  ⚠️ JSON parse error: {je}")
                        self.log(f"  Raw response (first 500 chars): {json_text[:500]}")
                    
                    # Retry with a fresh API call (up to 2 retries)
                    if retry_count < 2:
                        self.log(f"  🔄 Retrying API call ({retry_count + 1}/2)...")
                        time.sleep(2)  # Longer delay
                        # Get a fresh API key for retry
                        self.client = self._init_gemini_client()
                        return self._call_gemini_api(forename, retry_count + 1)
                    return None
            
        except Exception as e:
            self.log(f"  ❌ API error for '{forename}': {e}")
            # Retry on general errors too
            if retry_count < 2:
                self.log(f"  🔄 Retrying API call ({retry_count + 1}/2)...")
                time.sleep(2)
                self.client = self._init_gemini_client()
                return self._call_gemini_api(forename, retry_count + 1)
            return None
        
        return None
    
    def process_forename(self, forename_id: int, english_name: str) -> bool:
        """
        Process a single forename: call API and store results with all variants.
        
        Returns:
            True if successful, False if failed
        """
        # Update job status to in_progress
        self.conn.execute(
            """UPDATE forename_processing_jobs 
               SET status = 'in_progress', started_at = ?
               WHERE forename_id = ?""",
            (datetime.now().isoformat(), forename_id)
        )
        self.conn.commit()
        
        # Call Gemini API
        result = self._call_gemini_api(english_name)
        
        if not result:
            self.conn.execute(
                """UPDATE forename_processing_jobs 
                   SET status = 'failed', error_message = 'No response from API'
                   WHERE forename_id = ?""",
                (forename_id,)
            )
            self.conn.commit()
            return False
        
        try:
            # Update forename gender if provided
            gender = result.get('gender', '')
            if gender and gender in ('m', 'f'):
                self.conn.execute(
                    "UPDATE forenames SET gender = ? WHERE id = ?",
                    (gender, forename_id)
                )
            
            # Map compact field names to case names
            case_mapping = {
                'nominative': ('nom_full', 'nom_abbr'),
                'genitive': ('gen_full', 'gen_abbr'),
                'dative': ('dat_full', 'dat_abbr'),
                'accusative': ('acc_full', 'acc_abbr'),
                'ablative': ('abl_full', 'abl_abbr'),
            }
            
            total_variants = 0
            
            # Insert Latin forms from compact format
            for case_name, (full_key, abbr_key) in case_mapping.items():
                full_form = result.get(full_key, '')
                abbreviated = result.get(abbr_key, '')
                
                if full_form and abbreviated:
                    # Generate all variants using heuristics
                    variants = self.variant_gen.generate_all_variants(full_form, abbreviated)
                    
                    for full_var, abbrev_var, var_type in variants:
                        # Compute normalized form for lookup
                        normalized = self.variant_gen.normalize(abbrev_var)
                        
                        is_primary = 1 if var_type == 'primary' else 0
                        
                        try:
                            self.conn.execute(
                                """INSERT OR REPLACE INTO forename_latin_forms 
                                   (forename_id, case_name, latin_full, latin_abbreviated, 
                                    normalized_form, is_primary, variant_type)
                                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                                (forename_id, case_name, full_var, abbrev_var, 
                                 normalized, is_primary, var_type)
                            )
                            total_variants += 1
                        except sqlite3.IntegrityError:
                            # Duplicate - skip
                            pass
            
            # Update processing job
            self.conn.execute(
                """UPDATE forename_processing_jobs 
                   SET status = 'completed', 
                       completed_at = ?,
                       api_response = ?,
                       variants_generated = ?
                   WHERE forename_id = ?""",
                (datetime.now().isoformat(), json.dumps(result), total_variants, forename_id)
            )
            
            # Update forename processed_at
            self.conn.execute(
                "UPDATE forenames SET processed_at = ? WHERE id = ?",
                (datetime.now().isoformat(), forename_id)
            )
            
            self.conn.commit()
            return True
            
        except Exception as e:
            self.conn.execute(
                """UPDATE forename_processing_jobs 
                   SET status = 'failed', error_message = ?
                   WHERE forename_id = ?""",
                (str(e), forename_id)
            )
            self.conn.commit()
            return False
    
    def get_forenames_needing_variants(self) -> List[int]:
        """
        Find forenames that have been processed but are missing variant forms.
        
        A forename needs variant regeneration if:
        - It has completed processing (api_response stored)
        - It has no 'ij_variant' type entries (meaning it was processed before variant generation)
        
        Returns:
            List of forename IDs needing variant regeneration
        """
        query = """
            SELECT DISTINCT f.id
            FROM forenames f
            JOIN forename_processing_jobs j ON f.id = j.forename_id
            WHERE j.status = 'completed' 
              AND j.api_response IS NOT NULL
              AND f.id NOT IN (
                  SELECT DISTINCT forename_id 
                  FROM forename_latin_forms 
                  WHERE variant_type IN ('ij_variant', 'uv_variant', 'ij_uv_variant')
              )
        """
        cursor = self.conn.execute(query)
        return [row[0] for row in cursor.fetchall()]
    
    def auto_update_existing_entries(self) -> int:
        """
        Automatically detect and update existing entries that are missing variants.
        
        This is called automatically during processing to ensure all entries
        have complete variant coverage.
        
        Returns:
            Number of forenames updated
        """
        forenames_needing_update = self.get_forenames_needing_variants()
        
        if not forenames_needing_update:
            return 0
        
        self.log(f"Found {len(forenames_needing_update)} existing entries missing variants, updating...")
        
        updated = 0
        for fid in forenames_needing_update:
            new_variants = self.regenerate_variants(forename_id=fid)
            if new_variants > 0:
                updated += 1
        
        if updated > 0:
            self.log(f"Updated {updated} existing entries with new variants")
        
        return updated
    
    def regenerate_variants(self, forename_id: Optional[int] = None) -> int:
        """
        Regenerate orthographic variants for existing processed forenames.
        
        Useful after updating variant generation logic.
        
        Args:
            forename_id: Specific forename to regenerate, or None for all
            
        Returns:
            Number of new variants generated
        """
        if forename_id:
            query = """
                SELECT j.api_response, f.id, f.english_name
                FROM forename_processing_jobs j
                JOIN forenames f ON j.forename_id = f.id
                WHERE j.status = 'completed' AND f.id = ?
            """
            cursor = self.conn.execute(query, (forename_id,))
        else:
            query = """
                SELECT j.api_response, f.id, f.english_name
                FROM forename_processing_jobs j
                JOIN forenames f ON j.forename_id = f.id
                WHERE j.status = 'completed' AND j.api_response IS NOT NULL
            """
            cursor = self.conn.execute(query)
        
        total_new = 0
        
        for row in cursor.fetchall():
            api_response = row['api_response']
            fid = row['id']
            
            if not api_response:
                continue
            
            try:
                result = json.loads(api_response)
            except json.JSONDecodeError:
                continue
            
            case_mapping = {
                'nominative': ('nom_full', 'nom_abbr'),
                'genitive': ('gen_full', 'gen_abbr'),
                'dative': ('dat_full', 'dat_abbr'),
                'accusative': ('acc_full', 'acc_abbr'),
                'ablative': ('abl_full', 'abl_abbr'),
            }
            
            for case_name, (full_key, abbr_key) in case_mapping.items():
                full_form = result.get(full_key, '')
                abbreviated = result.get(abbr_key, '')
                
                if full_form and abbreviated:
                    variants = self.variant_gen.generate_all_variants(full_form, abbreviated)
                    
                    for full_var, abbrev_var, var_type in variants:
                        normalized = self.variant_gen.normalize(abbrev_var)
                        is_primary = 1 if var_type == 'primary' else 0
                        
                        try:
                            self.conn.execute(
                                """INSERT OR IGNORE INTO forename_latin_forms 
                                   (forename_id, case_name, latin_full, latin_abbreviated, 
                                    normalized_form, is_primary, variant_type)
                                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                                (fid, case_name, full_var, abbrev_var, 
                                 normalized, is_primary, var_type)
                            )
                            if self.conn.total_changes > 0:
                                total_new += 1
                        except sqlite3.IntegrityError:
                            pass
        
        self.conn.commit()
        return total_new
    
    def process_all_pending(self, limit: Optional[int] = None) -> Dict:
        """
        Process all pending forenames.
        
        Also automatically updates existing entries that are missing variants
        (e.g., entries processed before variant generation was added).
        
        Returns:
            Statistics about the processing run
        """
        self.initialize_processing_jobs()
        
        # Auto-update existing entries that are missing variants
        self.auto_update_existing_entries()
        
        pending = self.get_pending_forenames(limit)
        
        stats = {
            'total': len(pending),
            'processed': 0,
            'succeeded': 0,
            'failed': 0,
            'variants_generated': 0
        }
        
        self.log(f"Processing {len(pending)} forenames...")
        
        for i, forename in enumerate(pending, 1):
            forename_id = forename['id']
            english_name = forename['english_name']
            frequency = forename['frequency']
            
            self.log(f"  [{i}/{len(pending)}] {english_name} (freq: {frequency})")
            
            success = self.process_forename(forename_id, english_name)
            
            stats['processed'] += 1
            if success:
                stats['succeeded'] += 1
                variant_count = self._log_latin_forms(forename_id, english_name)
                stats['variants_generated'] += variant_count
            else:
                stats['failed'] += 1
                self.log(f"    ❌ Failed")
            
            # Delay between API calls
            if i < len(pending):
                time.sleep(self.delay)
        
        return stats
    
    def process_specific_forenames(self, forenames: List[str]) -> Dict:
        """Process specific forenames by name."""
        stats = {
            'total': len(forenames),
            'processed': 0,
            'succeeded': 0,
            'failed': 0,
            'not_found': 0,
            'variants_generated': 0
        }
        
        for forename in forenames:
            forename = forename.strip().title()
            
            # Check if exists, create if not
            cursor = self.conn.execute(
                "SELECT id FROM forenames WHERE english_name = ?",
                (forename,)
            )
            row = cursor.fetchone()
            
            if not row:
                # Create new forename entry
                cursor = self.conn.execute(
                    "INSERT INTO forenames (english_name, frequency) VALUES (?, 0)",
                    (forename,)
                )
                forename_id = cursor.lastrowid
                self.conn.commit()
                self.log(f"Created new forename entry: {forename}")
            else:
                forename_id = row['id']
            
            # Create processing job if needed
            self.conn.execute(
                """INSERT OR IGNORE INTO forename_processing_jobs (forename_id, status)
                   VALUES (?, 'pending')""",
                (forename_id,)
            )
            self.conn.commit()
            
            # Process
            self.log(f"Processing: {forename}")
            success = self.process_forename(forename_id, forename)
            
            stats['processed'] += 1
            if success:
                stats['succeeded'] += 1
                variant_count = self._log_latin_forms(forename_id, forename)
                stats['variants_generated'] += variant_count
            else:
                stats['failed'] += 1
                self.log(f"  ❌ Failed")
            
            time.sleep(self.delay)
        
        return stats
    
    def _log_latin_forms(self, forename_id: int, english_name: str) -> int:
        """
        Log the Latin forms for a successfully processed forename.
        
        Returns:
            Number of variants stored
        """
        cursor = self.conn.execute(
            """SELECT case_name, latin_full, latin_abbreviated, variant_type
               FROM forename_latin_forms
               WHERE forename_id = ?
               ORDER BY CASE case_name
                   WHEN 'nominative' THEN 1
                   WHEN 'genitive' THEN 2
                   WHEN 'dative' THEN 3
                   WHEN 'accusative' THEN 4
                   WHEN 'ablative' THEN 5
                   ELSE 6
               END,
               is_primary DESC,
               variant_type""",
            (forename_id,)
        )
        
        rows = cursor.fetchall()
        if not rows:
            self.log(f"    ✅ Completed (no forms stored)")
            return 0
        
        # Get gender
        cursor = self.conn.execute(
            "SELECT gender FROM forenames WHERE id = ?", (forename_id,)
        )
        gender_row = cursor.fetchone()
        gender = gender_row['gender'] if gender_row and gender_row['gender'] else '?'
        
        # Count variants by type
        primary_count = sum(1 for r in rows if r['variant_type'] == 'primary')
        variant_count = len(rows) - primary_count
        
        self.log(f"    ✅ {english_name} ({gender}): {primary_count} primary + {variant_count} variants")
        
        # Show primary forms only for brevity
        current_case = None
        for row in rows:
            if row['variant_type'] == 'primary':
                case_name = row['case_name']
                full = row['latin_full']
                abbrev = row['latin_abbreviated']
                if case_name != current_case:
                    self.log(f"       {case_name:11} {full:15} → {abbrev}")
                    current_case = case_name
        
        return len(rows)
    
    def lookup_latin_form(
        self, 
        text: str, 
        include_variants: bool = True
    ) -> List[Dict]:
        """
        Look up a Latin form using normalized matching.
        
        This is a robust lookup that will find matches regardless of:
        - I/J differences
        - U/V differences
        - Case differences
        - Abbreviation marker differences
        
        Args:
            text: Latin text to look up (can be full or abbreviated)
            include_variants: If True, also search variant forms
            
        Returns:
            List of matching forename records with their Latin forms
        """
        # Normalize the search text
        normalized = self.variant_gen.normalize(text)
        
        # Also search for the text as-is (case-insensitive)
        query = """
            SELECT DISTINCT 
                f.id as forename_id,
                f.english_name,
                f.gender,
                lf.case_name,
                lf.latin_full,
                lf.latin_abbreviated,
                lf.variant_type,
                lf.is_primary
            FROM forename_latin_forms lf
            JOIN forenames f ON lf.forename_id = f.id
            WHERE lf.normalized_form = ?
               OR LOWER(lf.latin_abbreviated) = LOWER(?)
               OR LOWER(lf.latin_full) = LOWER(?)
            ORDER BY f.frequency DESC, lf.is_primary DESC
        """
        
        cursor = self.conn.execute(query, (normalized, text, text))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'forename_id': row['forename_id'],
                'english_name': row['english_name'],
                'gender': row['gender'],
                'case_name': row['case_name'],
                'latin_full': row['latin_full'],
                'latin_abbreviated': row['latin_abbreviated'],
                'variant_type': row['variant_type'],
                'is_primary': bool(row['is_primary'])
            })
        
        return results
    
    def get_latin_forms(self, english_name: str, include_variants: bool = False) -> Optional[Dict]:
        """
        Look up Latin forms for a forename.
        
        Args:
            english_name: English forename to look up
            include_variants: If True, include all orthographic variants
        
        Returns:
            Dict with case -> list of {full, abbreviated, variant_type} mapping, or None if not found
        """
        if include_variants:
            query = """
                SELECT lf.case_name, lf.latin_full, lf.latin_abbreviated, 
                       lf.variant_type, lf.is_primary
                FROM forename_latin_forms lf
                JOIN forenames f ON lf.forename_id = f.id
                WHERE f.english_name = ?
                ORDER BY lf.case_name, lf.is_primary DESC, lf.variant_type
            """
        else:
            query = """
                SELECT lf.case_name, lf.latin_full, lf.latin_abbreviated,
                       lf.variant_type, lf.is_primary
                FROM forename_latin_forms lf
                JOIN forenames f ON lf.forename_id = f.id
                WHERE f.english_name = ? AND lf.is_primary = 1
                ORDER BY lf.case_name
            """
        
        cursor = self.conn.execute(query, (english_name.title(),))
        
        forms = {}
        for row in cursor.fetchall():
            case_name = row['case_name']
            if case_name not in forms:
                forms[case_name] = []
            forms[case_name].append({
                'full': row['latin_full'],
                'abbreviated': row['latin_abbreviated'],
                'variant_type': row['variant_type'],
                'is_primary': bool(row['is_primary'])
            })
        
        return forms if forms else None
    
    def show_stats(self):
        """Display statistics about forename processing."""
        print("\n" + "═" * 60)
        print("📊 FORENAME LATINIZATION STATISTICS")
        print("═" * 60)
        
        # Total forenames
        cursor = self.conn.execute("SELECT COUNT(*) FROM forenames")
        total = cursor.fetchone()[0]
        print(f"\n📝 Total unique forenames: {total:,}")
        
        # Processing status
        cursor = self.conn.execute(
            """SELECT status, COUNT(*) as count 
               FROM forename_processing_jobs 
               GROUP BY status"""
        )
        print("\n🔄 Processing Status:")
        for row in cursor.fetchall():
            emoji = {
                'completed': '✅',
                'pending': '⏳', 
                'failed': '❌',
                'in_progress': '🔄'
            }.get(row['status'], '❓')
            print(f"   {emoji} {row['status']}: {row['count']:,}")
        
        # Latin forms count
        cursor = self.conn.execute("SELECT COUNT(*) FROM forename_latin_forms")
        forms_count = cursor.fetchone()[0]
        print(f"\n📜 Total Latin forms stored: {forms_count:,}")
        
        # Variants by type
        cursor = self.conn.execute(
            """SELECT variant_type, COUNT(*) as count 
               FROM forename_latin_forms 
               GROUP BY variant_type
               ORDER BY count DESC"""
        )
        print("\n🔀 Forms by Variant Type:")
        for row in cursor.fetchall():
            var_type = row['variant_type'] or 'unknown'
            print(f"   {var_type}: {row['count']:,}")
        
        # Top forenames by frequency
        cursor = self.conn.execute(
            """SELECT english_name, frequency, gender
               FROM forenames 
               ORDER BY frequency DESC
               LIMIT 10"""
        )
        print("\n👤 Top 10 Forenames by Frequency:")
        for row in cursor.fetchall():
            gender = row['gender'] or '?'
            print(f"   {row['english_name']} ({gender}): {row['frequency']:,}")
        
        # Sample Latin forms with variants
        cursor = self.conn.execute(
            """SELECT f.english_name, 
                      COUNT(DISTINCT lf.latin_abbreviated) as variant_count
               FROM forename_latin_forms lf
               JOIN forenames f ON lf.forename_id = f.id
               WHERE lf.case_name = 'nominative'
               GROUP BY f.id
               ORDER BY f.frequency DESC
               LIMIT 5"""
        )
        print("\n📖 Sample Forenames with Variant Counts (Nominative):")
        for row in cursor.fetchall():
            print(f"   {row['english_name']}: {row['variant_count']} variants")
        
        print("\n" + "═" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Generate Latin declined/abbreviated forms for CP40 forenames',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract forenames and process all
  python cp40_forename_latinizer.py
  
  # Resume processing (skip completed)
  python cp40_forename_latinizer.py --resume
  
  # Process specific forenames
  python cp40_forename_latinizer.py --forenames "John,William,Thomas,Alice"
  
  # Show statistics only
  python cp40_forename_latinizer.py --stats
  
  # Limit number to process
  python cp40_forename_latinizer.py --limit 100
  
  # Lookup a Latin form (finds forename regardless of I/J, U/V)
  python cp40_forename_latinizer.py --lookup "Joh'es"
  python cp40_forename_latinizer.py --lookup "Iohannes"
  
  # Show all variants for an English forename
  python cp40_forename_latinizer.py --english "John" --all-variants
  
  # Regenerate variants for all processed forenames
  python cp40_forename_latinizer.py --regenerate-variants
  
  # Check and update existing entries missing variants
  python cp40_forename_latinizer.py --check-existing
        """
    )
    
    parser.add_argument(
        '--db',
        default='cp40_records.db',
        help='SQLite database path (default: cp40_records.db)'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=0.5,
        help='Delay between API calls in seconds (default: 0.5)'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show statistics and exit'
    )
    parser.add_argument(
        '--forenames',
        type=str,
        help='Comma-separated list of specific forenames to process'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume processing (skip already completed)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of forenames to process'
    )
    parser.add_argument(
        '--extract-only',
        action='store_true',
        help='Only extract forenames from persons table, do not call API'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )
    parser.add_argument(
        '--lookup',
        type=str,
        help='Look up a Latin form (robust matching with I/J, U/V normalization)'
    )
    parser.add_argument(
        '--english',
        type=str,
        help='Look up Latin forms for an English forename'
    )
    parser.add_argument(
        '--all-variants',
        action='store_true',
        help='With --english, show all orthographic variants'
    )
    parser.add_argument(
        '--regenerate-variants',
        action='store_true',
        help='Regenerate orthographic variants for all processed forenames'
    )
    parser.add_argument(
        '--check-existing',
        action='store_true',
        help='Check and update existing entries missing variants, then exit'
    )
    
    args = parser.parse_args()
    
    try:
        with ForenameLatinizer(
            db_path=args.db,
            delay=args.delay,
            verbose=not args.quiet
        ) as latinizer:
            
            if args.stats:
                latinizer.show_stats()
                return 0
            
            if args.lookup:
                results = latinizer.lookup_latin_form(args.lookup)
                if results:
                    print(f"\n🔍 Lookup results for '{args.lookup}':")
                    print(f"   (Normalized: '{LatinVariantGenerator.normalize(args.lookup)}')\n")
                    
                    # Group by forename
                    by_forename = {}
                    for r in results:
                        key = r['english_name']
                        if key not in by_forename:
                            by_forename[key] = []
                        by_forename[key].append(r)
                    
                    for english_name, forms in by_forename.items():
                        gender = forms[0]['gender'] or '?'
                        print(f"   {english_name} ({gender}):")
                        for f in forms:
                            primary = "★" if f['is_primary'] else " "
                            print(f"      {primary} {f['case_name']:11} {f['latin_full']:15} → {f['latin_abbreviated']} [{f['variant_type']}]")
                else:
                    print(f"No matches found for '{args.lookup}'")
                return 0
            
            if args.english:
                forms = latinizer.get_latin_forms(args.english, include_variants=args.all_variants)
                if forms:
                    print(f"\nLatin forms for '{args.english}':")
                    if args.all_variants:
                        print("(showing all orthographic variants)\n")
                    for case_name in LATIN_CASES:
                        if case_name in forms:
                            print(f"  {case_name}:")
                            for form in forms[case_name]:
                                primary = "★" if form['is_primary'] else " "
                                print(f"    {primary} {form['full']:15} → {form['abbreviated']} [{form['variant_type']}]")
                else:
                    print(f"No Latin forms found for '{args.english}'")
                return 0
            
            if args.regenerate_variants:
                latinizer.log("Regenerating orthographic variants for all processed forenames...")
                new_count = latinizer.regenerate_variants()
                latinizer.log(f"Generated {new_count} new variants")
                latinizer.show_stats()
                return 0
            
            if args.check_existing:
                missing = latinizer.get_forenames_needing_variants()
                if missing:
                    latinizer.log(f"Found {len(missing)} entries missing variants")
                    updated = latinizer.auto_update_existing_entries()
                    latinizer.log(f"Updated {updated} entries")
                else:
                    latinizer.log("All existing entries have variants - nothing to update")
                latinizer.show_stats()
                return 0
            
            # Extract forenames from persons table
            latinizer.extract_forenames_from_persons()
            
            if args.extract_only:
                latinizer.show_stats()
                return 0
            
            # Process forenames
            if args.forenames:
                forename_list = [f.strip() for f in args.forenames.split(',')]
                stats = latinizer.process_specific_forenames(forename_list)
            else:
                stats = latinizer.process_all_pending(limit=args.limit)
            
            # Show results
            print("\n" + "═" * 60)
            print("📈 PROCESSING SUMMARY")
            print("═" * 60)
            print(f"   Total: {stats['total']}")
            print(f"   Processed: {stats['processed']}")
            print(f"   Succeeded: {stats['succeeded']}")
            print(f"   Failed: {stats['failed']}")
            print(f"   Variants Generated: {stats.get('variants_generated', 0)}")
            
            latinizer.show_stats()
            
            return 0
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
