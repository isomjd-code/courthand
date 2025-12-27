#!/usr/bin/env python3
"""
Generate random text from Markov model and replace names/places with random database entries.

For each word in the generated text:
- If word appears in forename_latin_forms.latin_abbreviated, replace with random 
  latin_abbreviated from the same case_name
- If word appears in surnames.surname, replace with random surname
- If word appears in places.name, replace with random place name
"""

import json
import random
import sqlite3
import argparse
import time
import re
from pathlib import Path

try:
    import markovify
except ImportError:
    print("Error: markovify is not installed. Install it with: pip install markovify")
    raise

# Import the custom LatinText class
from build_markov_model import LatinText


def normalize_word(word: str) -> str:
    """
    Normalize a word by stripping trailing punctuation (except apostrophes which are part of abbreviations).
    This helps match words like "London," or "London." to "London" in the database.
    """
    # Strip trailing punctuation but keep apostrophes (they're part of abbreviations like "London'")
    # Remove trailing: comma, period, semicolon, colon, exclamation, question mark, etc.
    word = word.rstrip(',.;:!?')
    return word


def weighted_choice_fast(items, weights, exclude=None):
    """
    Choose a random item based on weights, excluding a specific item if provided.
    Optimized version that avoids creating new lists.
    
    Args:
        items: List of items
        weights: List of weights (same length as items)
        exclude: Item to exclude from selection (optional)
    
    Returns:
        Randomly selected item based on weights
    """
    if exclude:
        # Filter in-place without creating intermediate list
        filtered_items = []
        filtered_weights = []
        for item, weight in zip(items, weights):
            if item != exclude:
                filtered_items.append(item)
                filtered_weights.append(weight)
        if not filtered_items:
            return None
        items, weights = filtered_items, filtered_weights
    
    if not items:
        return None
    
    return random.choices(items, weights=weights, k=1)[0]


def load_model(model_path: Path):
    """Load the saved Markov model from JSON file."""
    print(f"Loading model from {model_path}...")
    
    with open(model_path, 'r', encoding='utf-8') as f:
        model_json_content = json.load(f)
    
    # Parse the JSON string
    if isinstance(model_json_content, str):
        model_dict = json.loads(model_json_content)
    else:
        model_dict = model_json_content
    
    # Reconstruct the model
    state_size = model_dict['state_size']
    chain = markovify.Chain.from_json(model_dict['chain'])
    parsed_sentences = model_dict['parsed_sentences']
    
    model = LatinText(chain, parsed_sentences, state_size)
    
    print("Model loaded successfully!")
    return model


class DatabaseReplacer:
    """
    Efficient database-backed word replacer that queries on-demand with caching.
    Uses minimal memory by only loading lookup sets and querying replacements as needed.
    """
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = None
        self._forename_set = None
        self._surname_set = None
        self._place_set = None
        self._place_abbreviated_set = None  # Set of abbreviated place forms for fast lookup
        self._place_latin_forms_map = None  # {latin_abbreviated: place_name}
        
        # LRU caches for frequently accessed replacements (max 1000 entries each)
        self._forename_cache = {}
        self._max_cache_size = 1000
        
        # Pre-computed weighted lists (only loaded once, shared across queries)
        self._surname_items = None
        self._surname_weights = None
        self._place_items = None
        self._place_weights = None
        
    def _connect(self):
        """Lazy connection to database."""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
    
    def _load_lookup_sets(self):
        """Load only the lookup sets (minimal memory footprint)."""
        if self._forename_set is not None:
            return
        
        self._connect()
        start_time = time.time()
        print(f"Loading lookup sets from database...")
        
        # Load forename lookup set (only unique words, not full mappings)
        try:
            cursor = self.conn.execute("""
                SELECT DISTINCT flf.latin_abbreviated
                FROM forename_latin_forms flf
                WHERE flf.latin_abbreviated IS NOT NULL AND flf.latin_abbreviated != ''
            """)
            self._forename_set = {row[0] for row in cursor.fetchall()}
            print(f"  Loaded {len(self._forename_set)} forename lookup entries")
        except sqlite3.OperationalError as e:
            print(f"  Warning: Could not load forename_latin_forms: {e}")
            self._forename_set = set()
        
        # Also check if we need to load place_latin_forms for abbreviated place names
        self._place_latin_forms_map = {}  # {latin_abbreviated: place_name}
        self._place_abbreviated_set = set()
        try:
            cursor = self.conn.execute("""
                SELECT plf.latin_abbreviated, p.name
                FROM place_latin_forms plf
                JOIN places p ON plf.place_id = p.id
                WHERE plf.latin_abbreviated IS NOT NULL AND plf.latin_abbreviated != ''
            """)
            for row in cursor.fetchall():
                latin_abbr = row[0]
                place_name = row[1]
                # Map abbreviated form to place name (keep first mapping if duplicates)
                if latin_abbr not in self._place_latin_forms_map:
                    self._place_latin_forms_map[latin_abbr] = place_name
                    self._place_abbreviated_set.add(latin_abbr)
            if self._place_latin_forms_map:
                print(f"  Loaded {len(self._place_latin_forms_map)} place Latin form mappings")
        except sqlite3.OperationalError:
            # place_latin_forms table might not exist, that's okay
            self._place_latin_forms_map = {}
            self._place_abbreviated_set = set()
        
        # Load surname lookup set and pre-compute weighted lists
        try:
            cursor = self.conn.execute("""
                SELECT s.surname, COUNT(DISTINCT ep.entry_id) as frequency
                FROM surnames s
                JOIN person_surnames ps ON s.id = ps.surname_id
                JOIN entry_persons ep ON ps.person_id = ep.person_id
                WHERE s.surname IS NOT NULL AND s.surname != ''
                GROUP BY s.id, s.surname
                HAVING frequency > 0
            """)
            rows = cursor.fetchall()
            self._surname_set = {row[0] for row in rows}
            # Pre-compute items and weights for fast weighted selection
            self._surname_items = [row[0] for row in rows]
            self._surname_weights = [row[1] for row in rows]
            print(f"  Loaded {len(self._surname_set)} surname lookup entries")
        except sqlite3.OperationalError as e:
            print(f"  Warning: Could not load surnames: {e}")
            self._surname_set = set()
            self._surname_items = []
            self._surname_weights = []
        
        # Load place lookup set and pre-compute weighted lists
        # Try multiple strategies to load places
        try:
            # Strategy 1: Try to use frequency column from places table if it exists
            try:
                cursor = self.conn.execute("""
                    SELECT p.name, COALESCE(p.frequency, 1) as frequency
                    FROM places p
                    WHERE p.name IS NOT NULL AND p.name != ''
                    ORDER BY p.name
                """)
                rows = cursor.fetchall()
                if rows:
                    self._place_set = {row[0] for row in rows}
                    self._place_items = [row[0] for row in rows]
                    self._place_weights = [max(1, row[1] or 1) for row in rows]
                    print(f"  Loaded {len(self._place_set)} place lookup entries (using places.frequency)")
                else:
                    raise sqlite3.OperationalError("No places found with frequency column")
            except (sqlite3.OperationalError, sqlite3.DatabaseError):
                # Strategy 2: Use LEFT JOIN to count from entry_places (includes places without entries)
                cursor = self.conn.execute("""
                    SELECT p.name, COALESCE(COUNT(epl.entry_id), 1) as frequency
                    FROM places p
                    LEFT JOIN entry_places epl ON p.id = epl.place_id
                    WHERE p.name IS NOT NULL AND p.name != ''
                    GROUP BY p.id, p.name
                """)
                rows = cursor.fetchall()
                if rows:
                    self._place_set = {row[0] for row in rows}
                    self._place_items = [row[0] for row in rows]
                    self._place_weights = [max(1, row[1] or 1) for row in rows]
                    print(f"  Loaded {len(self._place_set)} place lookup entries (using entry_places count)")
                else:
                    # Strategy 3: Just load all places with default frequency
                    cursor = self.conn.execute("""
                        SELECT p.name
                        FROM places p
                        WHERE p.name IS NOT NULL AND p.name != ''
                        ORDER BY p.name
                    """)
                    rows = cursor.fetchall()
                    self._place_set = {row[0] for row in rows}
                    self._place_items = [row[0] for row in rows]
                    self._place_weights = [1] * len(self._place_items)  # Default frequency of 1
                    print(f"  Loaded {len(self._place_set)} place lookup entries (default frequency)")
        except sqlite3.OperationalError as e:
            print(f"  Warning: Could not load places: {e}")
            self._place_set = set()
            self._place_items = []
            self._place_weights = []
        
        load_time = time.time() - start_time
        print(f"Lookup sets loaded in {load_time:.3f}s\n")
    
    def _get_forename_replacement(self, word: str) -> str:
        """Get a replacement forename for the given word, using cache."""
        # Check cache first
        if word in self._forename_cache:
            return self._forename_cache[word]
        
        # Query database for case_names containing this word
        cursor = self.conn.execute("""
            SELECT DISTINCT flf.case_name
            FROM forename_latin_forms flf
            WHERE flf.latin_abbreviated = ?
              AND flf.latin_abbreviated IS NOT NULL AND flf.latin_abbreviated != ''
        """, (word,))
        case_names = [row[0] for row in cursor.fetchall()]
        
        if not case_names:
            return None
        
        # Pick a random case_name
        case_name = random.choice(case_names)
        
        # Get replacements for this case_name (excluding the word itself)
        cursor = self.conn.execute("""
            SELECT flf.latin_abbreviated, COALESCE(f.frequency, 0) as frequency
            FROM forename_latin_forms flf
            JOIN forenames f ON flf.forename_id = f.id
            WHERE flf.case_name = ? 
              AND flf.latin_abbreviated IS NOT NULL 
              AND flf.latin_abbreviated != ''
              AND flf.latin_abbreviated != ?
        """, (case_name, word))
        rows = cursor.fetchall()
        
        if not rows:
            return None
        
        # Deduplicate and get max frequency per word
        word_freq = {}
        for row in rows:
            w, freq = row[0], row[1] or 0
            if w not in word_freq or freq > word_freq[w]:
                word_freq[w] = freq
        
        if not word_freq:
            return None
        
        # Weighted selection
        items = list(word_freq.keys())
        weights = [word_freq[w] for w in items]
        replacement = random.choices(items, weights=weights, k=1)[0]
        
        # Cache result (with size limit)
        if len(self._forename_cache) < self._max_cache_size:
            self._forename_cache[word] = replacement
        
        return replacement
    
    def _get_surname_replacement(self, word: str) -> str:
        """Get a replacement surname for the given word, using pre-computed lists."""
        if not self._surname_items:
            return None
        return weighted_choice_fast(self._surname_items, self._surname_weights, exclude=word)
    
    def _get_place_replacement(self, word: str) -> str:
        """Get a replacement place for the given word, using pre-computed lists."""
        if not self._place_items:
            return None
        return weighted_choice_fast(self._place_items, self._place_weights, exclude=word)
    
    def replace_words(self, text: str) -> tuple:
        """
        Replace words in text based on database mappings.
        Priority: forenames > surnames > places (first match wins)
        But if a word is in both forename/surname AND place sets, check place first for common place names.
        """
        self._load_lookup_sets()
        
        start_time = time.time()
        words = text.split()
        replacement_counts = {'forenames': 0, 'surnames': 0, 'places': 0}
        
        # Process words efficiently, building result as we go
        result_parts = []
        for word in words:
            replaced = False
            normalized = normalize_word(word)
            
            # Check if word (original or normalized) is in place set
            is_place = (word in self._place_set or normalized in self._place_set or 
                       (self._place_abbreviated_set and (word in self._place_abbreviated_set or normalized in self._place_abbreviated_set)))
            is_forename = normalized in self._forename_set
            is_surname = normalized in self._surname_set
            
            # If word is a place AND (forename or surname), prioritize place for common place names
            # Common place names that might also be names: London, York, Kent, etc.
            common_place_names = {'london', 'york', 'kent', 'suffolk', 'norfolk', 'essex', 
                                 'cambridge', 'oxford', 'canterbury', 'winchester', 'bristol'}
            
            if is_place and (is_forename or is_surname) and normalized.lower() in common_place_names:
                # Prioritize place for common place names
                place_name = None
                if word in self._place_set:
                    place_name = word
                elif normalized in self._place_set:
                    place_name = normalized
                elif self._place_abbreviated_set:
                    if word in self._place_abbreviated_set:
                        place_name = self._place_latin_forms_map.get(word)
                    elif normalized in self._place_abbreviated_set:
                        place_name = self._place_latin_forms_map.get(normalized)
                
                if place_name:
                    replacement = self._get_place_replacement(place_name)
                    if replacement:
                        result_parts.append(replacement)
                        replacement_counts['places'] += 1
                        replaced = True
            
            # Check forenames (if not already replaced as place)
            if not replaced and is_forename:
                replacement = self._get_forename_replacement(normalized)
                if replacement:
                    result_parts.append(replacement)
                    replacement_counts['forenames'] += 1
                    replaced = True
            
            # Check surnames (if not already replaced)
            if not replaced and is_surname:
                replacement = self._get_surname_replacement(normalized)
                if replacement:
                    result_parts.append(replacement)
                    replacement_counts['surnames'] += 1
                    replaced = True
            
            # Check places (both direct names and Latin abbreviated forms)
            if not replaced:
                place_name_to_check = None
                # Check original word first (might have apostrophe like "London'")
                if word in self._place_set:
                    place_name_to_check = word
                elif normalized in self._place_set:
                    place_name_to_check = normalized
                # Check if word (original or normalized) is an abbreviated form in place_latin_forms
                elif self._place_abbreviated_set:
                    if word in self._place_abbreviated_set:
                        place_name_to_check = self._place_latin_forms_map.get(word)
                    elif normalized in self._place_abbreviated_set:
                        place_name_to_check = self._place_latin_forms_map.get(normalized)
                
                if place_name_to_check:
                    replacement = self._get_place_replacement(place_name_to_check)
                    if replacement:
                        result_parts.append(replacement)
                        replacement_counts['places'] += 1
                        replaced = True
            
            # Keep original if no replacement
            if not replaced:
                result_parts.append(word)
        
        result = ' '.join(result_parts)
        total_time = time.time() - start_time
        print(f"Replacement time: {total_time:.3f}s\n")
        
        return result, replacement_counts
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None


def main():
    parser = argparse.ArgumentParser(description='Generate text with database name/place replacements')
    parser.add_argument('--model', type=str, default='markov_model.json',
                       help='Path to the saved Markov model JSON file')
    parser.add_argument('--db', type=str, default='cp40_records.db',
                       help='Path to the database file')
    parser.add_argument('--tries', type=int, default=50,
                       help='Maximum attempts to generate a sentence (default: 50)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
    
    # Load model
    model_path = Path(__file__).parent / args.model
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return
    
    model = load_model(model_path)
    
    # Initialize database replacer (lazy loading, minimal memory)
    db_path = Path(__file__).parent / args.db
    if not db_path.exists():
        print(f"Error: Database file not found: {db_path}")
        return
    
    replacer = DatabaseReplacer(db_path)
    
    try:
        # Generate text
        print(f"Generating text...")
        generate_start = time.time()
        sentence = None
        for attempt in range(args.tries):
            try:
                sentence = model.make_sentence()
                if sentence:
                    break
            except Exception as e:
                if attempt == args.tries - 1:
                    print(f"Error: Failed to generate sentence: {e}")
                    return
                continue
        generate_time = time.time() - generate_start
        print(f"Text generation: {generate_time:.3f}s\n")
        
        if not sentence:
            print("Error: Could not generate a sentence")
            return
        
        print(f"Original text: {sentence}\n")
        
        # Replace words (database queries happen on-demand)
        replaced_text, counts = replacer.replace_words(sentence)
        
        print(f"Replaced text: {replaced_text}\n")
        print(f"Replacements made:")
        print(f"  Forenames: {counts['forenames']}")
        print(f"  Surnames: {counts['surnames']}")
        print(f"  Places: {counts['places']}")
        print(f"  Total: {sum(counts.values())}")
    finally:
        replacer.close()


if __name__ == "__main__":
    main()
