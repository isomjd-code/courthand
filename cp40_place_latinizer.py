#!/usr/bin/env python3
"""
CP40 Place Name Latinizer
Extracts place names from the CP40 database and generates Latin forms using Gemini 3 Flash.
Processes 20 entries at a time to obtain latinized forms as they might appear in CP40 case transcriptions.

Usage:
    # Extract place names and populate Latin forms
    python cp40_place_latinizer.py
    
    # Show statistics only
    python cp40_place_latinizer.py --stats
    
    # Process specific place names
    python cp40_place_latinizer.py --places "London,York,Canterbury"
    
    # Resume processing (skip already completed)
    python cp40_place_latinizer.py --resume
    
    # Look up all variants for a place
    python cp40_place_latinizer.py --lookup "Londinium"
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
    Randomly selects from a pool of keys to distribute load.
    
    Returns:
        str: A free API key for Gemini API
    """
    seed = random.randint(1, 4)
    if seed == 1:
        return 'AIzaSyCKaa2wRfvl2Tdm54z4UndljxaWAF0AT3s'
    elif seed == 2:
        return 'AIzaSyAuYGtV_gHv3MRtvjX_4XsheSgA9-Yfv88'
    elif seed == 3:
        return 'AIzaSyDHFVLSQOAuVgbgPdHSXGQ4szcTxO9kK9s'
    elif seed == 4:
        return 'AIzaSyBtAL-otqjPgw6Vu6Lyba3-22K0N-f3-k'
    else:
        return 'AIzaSyCKaa2wRfvl2Tdm54z4UndljxaWAF0AT3s'


# Base schema for place Latin forms
PLACE_SCHEMA_SQL = """
-- ═══════════════════════════════════════════════════════════════
-- PLACES TABLE (already exists, but we'll ensure it's compatible)
-- ═══════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS places (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    frequency INTEGER DEFAULT 0,
    processed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ═══════════════════════════════════════════════════════════════
-- LATIN FORMS TABLE (Latin versions of place names)
-- Each place can have multiple Latin forms (full and abbreviated)
-- ═══════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS place_latin_forms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    place_id INTEGER NOT NULL REFERENCES places(id) ON DELETE CASCADE,
    latin_full TEXT NOT NULL,  -- Full Latin form (e.g., "Londinium" or "Londinium")
    latin_abbreviated TEXT NOT NULL,  -- Abbreviated form with apostrophe (e.g., "Lond'ium")
    is_primary BOOLEAN DEFAULT 0,  -- Primary/canonical form from LLM
    normalized_form TEXT,  -- Normalized for lookup (I/J, U/V normalized)
    variant_type TEXT DEFAULT 'primary',  -- primary, ij_variant, uv_variant, abbrev_variant, etc.
    notes TEXT,  -- Any special notes about this form
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(place_id, latin_abbreviated)
);

-- ═══════════════════════════════════════════════════════════════
-- PROCESSING STATUS TABLE (track API call progress)
-- ═══════════════════════════════════════════════════════════════
CREATE TABLE IF NOT EXISTS place_processing_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    place_id INTEGER NOT NULL REFERENCES places(id) ON DELETE CASCADE,
    status TEXT DEFAULT 'pending',  -- pending, in_progress, completed, failed
    error_message TEXT,
    api_response TEXT,  -- Store raw API response for debugging
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    variants_generated INTEGER DEFAULT 0,
    UNIQUE(place_id)
);

-- ═══════════════════════════════════════════════════════════════
-- INDEXES
-- ═══════════════════════════════════════════════════════════════
CREATE INDEX IF NOT EXISTS idx_places_name ON places(name);
CREATE INDEX IF NOT EXISTS idx_latin_forms_place ON place_latin_forms(place_id);
CREATE INDEX IF NOT EXISTS idx_latin_forms_abbreviated ON place_latin_forms(latin_abbreviated);
CREATE INDEX IF NOT EXISTS idx_latin_forms_normalized ON place_latin_forms(normalized_form);
CREATE INDEX IF NOT EXISTS idx_processing_status ON place_processing_jobs(status);
"""


class LatinVariantGenerator:
    """
    Generates orthographic variants for Latin place names using reliable heuristics.
    
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
        
        # V → U normalization
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
    def generate_abbreviation_variants(full_form: str, primary_abbrev: str) -> Set[str]:
        """
        Generate multiple plausible abbreviation variants for place names.
        
        Medieval abbreviation patterns:
        1. Contraction with apostrophe: Lond'ium (Londinium)
        2. Suspension (truncation): Lond' or Lon.
        3. First + last syllable: Lond...ium
        
        Args:
            full_form: Full Latin form (e.g., "Londinium")
            primary_abbrev: Primary abbreviation from LLM (e.g., "Lond'ium")
            
        Returns:
            Set of plausible abbreviation variants
        """
        if not full_form:
            return {primary_abbrev} if primary_abbrev else set()
        
        variants = {primary_abbrev} if primary_abbrev else set()
        full_lower = full_form.lower()
        
        # Common abbreviation patterns based on word structure
        if len(full_form) >= 4:
            # Get the ending (typically last 2-3 chars)
            endings = []
            if full_lower.endswith(('um', 'us', 'ia', 'is', 'ae', 'am')):
                endings.append(full_form[-2:])
            if full_lower.endswith(('ium', 'ium', 'orum', 'arum')):
                endings.append(full_form[-3:])
            if full_lower.endswith(('i', 'o', 'e', 'a', 'u')):
                endings.append(full_form[-1:])
            
            # Common prefix lengths to try
            prefix_lengths = [3, 4, 5, 6]
            
            for prefix_len in prefix_lengths:
                if prefix_len < len(full_form):
                    prefix = full_form[:prefix_len]
                    for ending in endings:
                        # Make sure we're actually abbreviating
                        if len(prefix) + len(ending) + 1 < len(full_form):
                            abbrev = f"{prefix}'{ending}"
                            variants.add(abbrev)
            
            # Also try prefix + apostrophe + longer endings
            for stem_len in range(3, min(len(full_form) - 2, 7)):
                stem = full_form[:stem_len]
                remaining = full_form[stem_len:]
                if len(remaining) >= 2:
                    abbrev = f"{stem}'{remaining}"
                    if abbrev != full_form and len(abbrev) < len(full_form):
                        variants.add(abbrev)
        
        # Single apostrophe at end (suspension)
        for i in range(3, min(7, len(full_form))):
            variants.add(full_form[:i] + "'")
        
        # Add the full unabbreviated form
        if full_form and len(full_form) >= 3:
            variants.add(full_form)
        
        # Add suspension variants that cut off just the ending
        if len(full_form) >= 4:
            for cut_len in [1, 2, 3]:
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
        Generate all orthographic variants for a Latin place name.
        
        Args:
            full_form: Full Latin form (e.g., "Londinium")
            abbreviated: Primary abbreviated form (e.g., "Lond'ium")
            
        Returns:
            List of tuples: (full_variant, abbrev_variant, variant_type)
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
            # Determine variant type
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
            abbrev_var = abbreviated
            if has_ij_diff:
                ij_abbrevs = cls.generate_ij_variants(abbreviated)
                for ij_abbr in ij_abbrevs:
                    if ij_abbr != abbreviated:
                        results.append((full_var, ij_abbr, var_type))
                abbrev_var = list(ij_abbrevs)[0] if ij_abbrevs else abbreviated
            
            if has_uv_diff:
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
        
        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for item in results:
            key = (item[0], item[1])
            if key not in seen:
                seen.add(key)
                deduped.append(item)
        
        return deduped


class PlaceLatinizer:
    """
    Extracts place names from CP40 database and generates Latin forms using Gemini,
    then expands with heuristic-based orthographic variants.
    """
    
    def __init__(
        self, 
        db_path: str = "cp40_records.db",
        delay: float = 0.5,
        verbose: bool = True,
        batch_size: int = 20
    ):
        """
        Initialize the latinizer.
        
        Args:
            db_path: Path to the CP40 SQLite database
            delay: Delay between API calls in seconds
            verbose: Print progress information
            batch_size: Number of places to process per API call
        """
        self.db_path = db_path
        self.delay = delay
        self.verbose = verbose
        self.batch_size = batch_size
        
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
        """Initialize Gemini API client using paid API key."""
        api_key = "AIzaSyBmFe4P5cV1L7L5EmjLVC32AQiTQHmgJ7A"
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable must be set with a paid API key")
        return genai.Client(api_key=api_key)
    
    def _create_schema(self):
        """Create the place schema if it doesn't exist."""
        self.conn.executescript(PLACE_SCHEMA_SQL)
        self.conn.commit()
    
    def _run_migrations(self):
        """Run database migrations for new columns."""
        migrations_run = []
        
        # Check places table columns
        cursor = self.conn.execute("PRAGMA table_info(places)")
        places_columns = {row['name'] for row in cursor.fetchall()}
        
        # Add processed_at column to places if missing
        if 'processed_at' not in places_columns:
            try:
                self.conn.execute(
                    "ALTER TABLE places ADD COLUMN processed_at TIMESTAMP"
                )
                migrations_run.append("Added processed_at column to places")
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise
        
        # Add frequency column to places if missing
        if 'frequency' not in places_columns:
            try:
                self.conn.execute(
                    "ALTER TABLE places ADD COLUMN frequency INTEGER DEFAULT 0"
                )
                migrations_run.append("Added frequency column to places")
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise
        
        # Check place_latin_forms columns
        cursor = self.conn.execute("PRAGMA table_info(place_latin_forms)")
        latin_forms_columns = {row['name'] for row in cursor.fetchall()}
        
        # Add normalized_form column if missing
        if 'normalized_form' not in latin_forms_columns:
            try:
                self.conn.execute(
                    "ALTER TABLE place_latin_forms ADD COLUMN normalized_form TEXT"
                )
                migrations_run.append("Added normalized_form column")
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise
        
        # Add variant_type column if missing
        if 'variant_type' not in latin_forms_columns:
            try:
                self.conn.execute(
                    "ALTER TABLE place_latin_forms ADD COLUMN variant_type TEXT DEFAULT 'primary'"
                )
                migrations_run.append("Added variant_type column")
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise
        
        # Check processing_jobs table
        cursor = self.conn.execute("PRAGMA table_info(place_processing_jobs)")
        job_columns = {row['name'] for row in cursor.fetchall()}
        
        if 'variants_generated' not in job_columns:
            try:
                self.conn.execute(
                    "ALTER TABLE place_processing_jobs ADD COLUMN variants_generated INTEGER DEFAULT 0"
                )
                migrations_run.append("Added variants_generated column")
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise
        
        self.conn.commit()
        
        # Create indexes
        if 'normalized_form' in latin_forms_columns or 'normalized_form' in self.conn.execute("PRAGMA table_info(place_latin_forms)").fetchall():
            try:
                self.conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_latin_forms_normalized ON place_latin_forms(normalized_form)"
                )
            except sqlite3.OperationalError:
                pass
        
        if 'variant_type' in latin_forms_columns or 'variant_type' in self.conn.execute("PRAGMA table_info(place_latin_forms)").fetchall():
            try:
                self.conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_latin_forms_variant_type ON place_latin_forms(variant_type)"
                )
            except sqlite3.OperationalError:
                pass
        
        # Backfill normalized_form for existing records
        cursor = self.conn.execute("PRAGMA table_info(place_latin_forms)")
        latin_forms_columns = {row['name'] for row in cursor.fetchall()}
        if 'normalized_form' in latin_forms_columns:
            self.conn.execute("""
                UPDATE place_latin_forms 
                SET normalized_form = LOWER(REPLACE(REPLACE(REPLACE(latin_abbreviated, 'j', 'i'), 'J', 'I'), '''', ''))
                WHERE normalized_form IS NULL OR normalized_form = ''
            """)
        
        # Backfill variant_type for existing records
        if 'variant_type' in latin_forms_columns:
            self.conn.execute("""
                UPDATE place_latin_forms 
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
    
    def initialize_processing_jobs(self):
        """Create processing jobs for all places that don't have one."""
        self.conn.execute("""
            INSERT OR IGNORE INTO place_processing_jobs (place_id, status)
            SELECT id, 'pending' FROM places
            WHERE id NOT IN (SELECT place_id FROM place_processing_jobs)
        """)
        self.conn.commit()
    
    def get_pending_places(self, limit: Optional[int] = None) -> List[Dict]:
        """Get places that haven't been processed yet."""
        query = """
            SELECT p.id, p.name
            FROM places p
            JOIN place_processing_jobs j ON p.id = j.place_id
            WHERE j.status IN ('pending', 'failed')
            ORDER BY p.id
        """
        if limit:
            query += f" LIMIT {limit}"
        
        cursor = self.conn.execute(query)
        return [dict(row) for row in cursor.fetchall()]
    
    def _build_prompt(self, place_names: List[str]) -> str:
        """Build the prompt for Gemini to generate Latin forms for a batch of places."""
        names_list = '", "'.join(place_names)
        return f"""Medieval Latin forms for these English place names as they might appear in CP40 case transcriptions: "{names_list}".

Rules:
- Use straight apostrophe (') for abbreviated/dropped letters
- Use J (not I) for initial J sounds - we will generate I variants automatically
- Place names in medieval Latin documents are typically in locative or ablative case
- Examples: Londinium → Lond'ium, Eboracum → Ebor'acum, Cantuaria → Cant'aria

Return ONLY a JSON array with one object per place name (no extra whitespace):
[
  {{"name":"{place_names[0]}","latin_full":"","latin_abbr":""}},
  {{"name":"{place_names[1] if len(place_names) > 1 else place_names[0]}","latin_full":"","latin_abbr":""}}
]

Fill in all fields. Example for London:
{{"name":"London","latin_full":"Londinium","latin_abbr":"Lond'ium"}}"""
    
    def _call_gemini_api(self, place_names: List[str], retry_count: int = 0) -> Optional[List[Dict]]:
        """
        Call Gemini 3 Flash to get Latin forms for a batch of place names.
        
        Returns:
            List of parsed JSON objects or None if failed
        """
        prompt = self._build_prompt(place_names)
        
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
                
                # Try to parse as JSON array
                try:
                    parsed = json.loads(json_text)
                    if isinstance(parsed, list):
                        return parsed
                    elif isinstance(parsed, dict):
                        # Single object, wrap in list
                        return [parsed]
                except json.JSONDecodeError:
                    # Try to extract JSON objects from text
                    json_objects = re.findall(r'\{[^{}]*\}', json_text)
                    parsed_objects = []
                    for obj_str in json_objects:
                        try:
                            parsed_obj = json.loads(obj_str)
                            parsed_objects.append(parsed_obj)
                        except json.JSONDecodeError:
                            continue
                    
                    if parsed_objects:
                        return parsed_objects
                    
                    # Log the raw response for debugging
                    if self.verbose:
                        self.log(f"  ⚠️ JSON parse error")
                        self.log(f"  Raw response (first 500 chars): {json_text[:500]}")
                    
                    # Retry with a fresh API call (up to 2 retries)
                    if retry_count < 2:
                        self.log(f"  🔄 Retrying API call ({retry_count + 1}/2)...")
                        time.sleep(2)  # Longer delay
                        self.client = self._init_gemini_client()
                        return self._call_gemini_api(place_names, retry_count + 1)
                    return None
            
        except Exception as e:
            self.log(f"  ❌ API error for batch: {e}")
            # Retry on general errors too
            if retry_count < 2:
                self.log(f"  🔄 Retrying API call ({retry_count + 1}/2)...")
                time.sleep(2)
                self.client = self._init_gemini_client()
                return self._call_gemini_api(place_names, retry_count + 1)
            return None
        
        return None
    
    def process_batch(self, place_batch: List[Dict]) -> int:
        """
        Process a batch of places: call API and store results with all variants.
        
        Args:
            place_batch: List of dicts with 'id' and 'name' keys
            
        Returns:
            Number of successfully processed places
        """
        place_names = [p['name'] for p in place_batch]
        place_ids = {p['name']: p['id'] for p in place_batch}
        
        # Update job statuses to in_progress
        place_id_list = [p['id'] for p in place_batch]
        placeholders = ','.join(['?'] * len(place_id_list))
        self.conn.execute(
            f"""UPDATE place_processing_jobs 
               SET status = 'in_progress', started_at = ?
               WHERE place_id IN ({placeholders})""",
            [datetime.now().isoformat()] + place_id_list
        )
        self.conn.commit()
        
        # Call Gemini API
        results = self._call_gemini_api(place_names)
        
        if not results:
            # Mark all as failed
            self.conn.execute(
                f"""UPDATE place_processing_jobs 
                   SET status = 'failed', error_message = 'No response from API'
                   WHERE place_id IN ({placeholders})""",
                place_id_list
            )
            self.conn.commit()
            return 0
        
        # Process results
        success_count = 0
        total_variants = 0
        
        # Create a mapping from name to result
        results_by_name = {}
        for result in results:
            name = result.get('name', '').strip()
            if name:
                results_by_name[name] = result
        
        for place in place_batch:
            place_id = place['id']
            place_name = place['name']
            
            # Find matching result (try exact match first, then case-insensitive)
            result = results_by_name.get(place_name)
            if not result:
                # Try case-insensitive match
                for key, val in results_by_name.items():
                    if key.lower() == place_name.lower():
                        result = val
                        break
            
            if not result:
                # No matching result found
                self.conn.execute(
                    """UPDATE place_processing_jobs 
                       SET status = 'failed', error_message = 'No matching result in API response'
                       WHERE place_id = ?""",
                    (place_id,)
                )
                continue
            
            try:
                latin_full = result.get('latin_full', '').strip()
                latin_abbr = result.get('latin_abbr', '').strip()
                
                if not latin_full or not latin_abbr:
                    self.conn.execute(
                        """UPDATE place_processing_jobs 
                           SET status = 'failed', error_message = 'Missing latin_full or latin_abbr in response'
                           WHERE place_id = ?""",
                        (place_id,)
                    )
                    continue
                
                # Generate all variants using heuristics
                variants = self.variant_gen.generate_all_variants(latin_full, latin_abbr)
                
                batch_variants = 0
                for full_var, abbrev_var, var_type in variants:
                    # Compute normalized form for lookup
                    normalized = self.variant_gen.normalize(abbrev_var)
                    
                    is_primary = 1 if var_type == 'primary' else 0
                    
                    try:
                        self.conn.execute(
                            """INSERT OR REPLACE INTO place_latin_forms 
                               (place_id, latin_full, latin_abbreviated, 
                                normalized_form, is_primary, variant_type)
                               VALUES (?, ?, ?, ?, ?, ?)""",
                            (place_id, full_var, abbrev_var, 
                             normalized, is_primary, var_type)
                        )
                        batch_variants += 1
                    except sqlite3.IntegrityError:
                        # Duplicate - skip
                        pass
                
                total_variants += batch_variants
                
                # Update processing job
                self.conn.execute(
                    """UPDATE place_processing_jobs 
                       SET status = 'completed', 
                           completed_at = ?,
                           api_response = ?,
                           variants_generated = ?
                       WHERE place_id = ?""",
                    (datetime.now().isoformat(), json.dumps(result), batch_variants, place_id)
                )
                
                # Update place processed_at (if column exists)
                try:
                    self.conn.execute(
                        "UPDATE places SET processed_at = ? WHERE id = ?",
                        (datetime.now().isoformat(), place_id)
                    )
                except sqlite3.OperationalError:
                    # Column might not exist, skip
                    pass
                
                success_count += 1
                
            except Exception as e:
                self.conn.execute(
                    """UPDATE place_processing_jobs 
                       SET status = 'failed', error_message = ?
                       WHERE place_id = ?""",
                    (str(e), place_id)
                )
        
        self.conn.commit()
        return success_count
    
    def process_all_pending(self, limit: Optional[int] = None) -> Dict:
        """
        Process all pending places in batches of batch_size.
        
        Returns:
            Statistics about the processing run
        """
        self.initialize_processing_jobs()
        
        pending = self.get_pending_places(limit)
        
        stats = {
            'total': len(pending),
            'processed': 0,
            'succeeded': 0,
            'failed': 0,
            'variants_generated': 0,
            'batches': 0
        }
        
        self.log(f"Processing {len(pending)} places in batches of {self.batch_size}...")
        
        # Process in batches
        for i in range(0, len(pending), self.batch_size):
            batch = pending[i:i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            total_batches = (len(pending) + self.batch_size - 1) // self.batch_size
            
            self.log(f"  Batch {batch_num}/{total_batches}: Processing {len(batch)} places...")
            for place in batch:
                self.log(f"    - {place['name']}")
            
            success_count = self.process_batch(batch)
            
            stats['batches'] += 1
            stats['processed'] += len(batch)
            stats['succeeded'] += success_count
            stats['failed'] += (len(batch) - success_count)
            
            # Get variant count for successfully processed places in this batch
            if success_count > 0:
                # Get variant count from the processing jobs for this batch
                place_ids = [p['id'] for p in batch]
                placeholders = ','.join(['?'] * len(place_ids))
                cursor = self.conn.execute(
                    f"""SELECT SUM(variants_generated) FROM place_processing_jobs 
                        WHERE place_id IN ({placeholders}) AND status = 'completed'""",
                    place_ids
                )
                result = cursor.fetchone()
                batch_variants = result[0] if result[0] else 0
                stats['variants_generated'] += batch_variants
                self.log(f"    ✅ Batch {batch_num} completed: {success_count}/{len(batch)} succeeded, {batch_variants} variants")
            else:
                self.log(f"    ❌ Batch {batch_num} completed: {success_count}/{len(batch)} succeeded")
            
            # Delay between API calls
            if i + self.batch_size < len(pending):
                time.sleep(self.delay)
        
        return stats
    
    def show_stats(self):
        """Display statistics about place processing."""
        print("\n" + "═" * 60)
        print("📊 PLACE NAME LATINIZATION STATISTICS")
        print("═" * 60)
        
        # Total places
        cursor = self.conn.execute("SELECT COUNT(*) FROM places")
        total = cursor.fetchone()[0]
        print(f"\n📝 Total unique places: {total:,}")
        
        # Processing status
        cursor = self.conn.execute(
            """SELECT status, COUNT(*) as count 
               FROM place_processing_jobs 
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
        cursor = self.conn.execute("SELECT COUNT(*) FROM place_latin_forms")
        forms_count = cursor.fetchone()[0]
        print(f"\n📜 Total Latin forms stored: {forms_count:,}")
        
        # Variants by type
        cursor = self.conn.execute(
            """SELECT variant_type, COUNT(*) as count 
               FROM place_latin_forms 
               GROUP BY variant_type
               ORDER BY count DESC"""
        )
        print("\n🔀 Forms by Variant Type:")
        for row in cursor.fetchall():
            var_type = row['variant_type'] or 'unknown'
            print(f"   {var_type}: {row['count']:,}")
        
        # Sample places
        cursor = self.conn.execute(
            """SELECT p.name, 
                      COUNT(DISTINCT lf.latin_abbreviated) as variant_count
               FROM place_latin_forms lf
               JOIN places p ON lf.place_id = p.id
               GROUP BY p.id
               ORDER BY variant_count DESC
               LIMIT 10"""
        )
        print("\n📖 Top 10 Places by Variant Count:")
        for row in cursor.fetchall():
            print(f"   {row['name']}: {row['variant_count']} variants")
        
        print("\n" + "═" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Generate Latin forms for CP40 place names',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all places in batches of 20
  python cp40_place_latinizer.py
  
  # Resume processing (skip completed)
  python cp40_place_latinizer.py --resume
  
  # Process specific places
  python cp40_place_latinizer.py --places "London,York,Canterbury"
  
  # Show statistics only
  python cp40_place_latinizer.py --stats
  
  # Limit number to process
  python cp40_place_latinizer.py --limit 100
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
        '--batch-size',
        type=int,
        default=20,
        help='Number of places to process per API call (default: 20)'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show statistics and exit'
    )
    parser.add_argument(
        '--places',
        type=str,
        help='Comma-separated list of specific place names to process'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume processing (skip already completed)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of places to process'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )
    
    args = parser.parse_args()
    
    try:
        with PlaceLatinizer(
            db_path=args.db,
            delay=args.delay,
            verbose=not args.quiet,
            batch_size=args.batch_size
        ) as latinizer:
            
            if args.stats:
                latinizer.show_stats()
                return 0
            
            # Process places
            if args.places:
                place_list = [p.strip() for p in args.places.split(',')]
                # Create a batch for these specific places
                place_batch = []
                for place_name in place_list:
                    cursor = latinizer.conn.execute(
                        "SELECT id, name FROM places WHERE name = ?",
                        (place_name,)
                    )
                    row = cursor.fetchone()
                    if row:
                        place_batch.append(dict(row))
                    else:
                        latinizer.log(f"Place '{place_name}' not found in database")
                
                if place_batch:
                    success_count = latinizer.process_batch(place_batch)
                    latinizer.log(f"Processed {success_count}/{len(place_batch)} places")
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
                print(f"   Batches: {stats['batches']}")
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

