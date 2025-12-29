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
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Set, FrozenSet

try:
    import markovify
except ImportError:
    print("Error: markovify is not installed. Install it with: pip install markovify")
    raise

# Import tqdm with fallback
try:
    from tqdm import tqdm
except (ImportError, ModuleNotFoundError):
    def tqdm(iterable, *args, **kwargs):
        return iterable

# Import the custom LatinText class
from build_markov_model import LatinText

# Punctuation to strip from word endings (compile once)
TRAILING_PUNCTUATION = ',.;:!?'


def normalize_word(word: str) -> str:
    """
    Normalize a word by stripping trailing punctuation (except apostrophes which are part of abbreviations).
    """
    return word.rstrip(TRAILING_PUNCTUATION)


def weighted_choice_excluding(
    items: List[str],
    weights: List[int],
    exclude: Optional[str] = None
) -> Optional[str]:
    """
    Choose a random item based on weights, optionally excluding a specific item.
    Uses random.choices which is optimized for weighted selection.
    """
    if not items:
        return None
    
    if exclude is None:
        return random.choices(items, weights=weights, k=1)[0]
    
    # Build filtered lists only when exclusion is needed
    filtered_items = []
    filtered_weights = []
    for item, weight in zip(items, weights):
        if item != exclude:
            filtered_items.append(item)
            filtered_weights.append(weight)
    
    if not filtered_items:
        return None
    
    return random.choices(filtered_items, weights=filtered_weights, k=1)[0]


def load_model(model_path: Path) -> LatinText:
    """Load the saved Markov model from JSON file."""
    print(f"Loading model from {model_path}...")
    
    with open(model_path, 'r', encoding='utf-8') as f:
        model_json_content = json.load(f)
    
    model_dict = json.loads(model_json_content) if isinstance(model_json_content, str) else model_json_content
    
    state_size = model_dict['state_size']
    chain = markovify.Chain.from_json(model_dict['chain'])
    parsed_sentences = model_dict['parsed_sentences']
    
    model = LatinText(chain, parsed_sentences, state_size)
    print("Model loaded successfully!")
    return model


class WordType:
    """Enum-like constants for word types."""
    FORENAME = 'forename'
    SURNAME = 'surname'
    PLACE = 'place'
    PLACE_ABBR = 'place_abbr'


class DatabaseReplacer:
    """
    Efficient database-backed word replacer with caching and lazy loading.
    """
    
    # Common place names that should be prioritized over name matches
    COMMON_PLACE_NAMES: FrozenSet[str] = frozenset({
        'london', 'york', 'kent', 'suffolk', 'norfolk', 'essex',
        'cambridge', 'oxford', 'canterbury', 'winchester', 'bristol'
    })
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self._initialized = False
        
        # Lookup sets
        self._forename_set: Set[str] = set()
        self._surname_set: Set[str] = set()
        self._place_set: Set[str] = set()
        self._place_abbreviated_set: Set[str] = set()
        
        # Mappings
        self._place_latin_forms_map: Dict[str, str] = {}
        self._forename_cases: Dict[str, List[Tuple[str, int]]] = {}
        self._word_to_cases: Dict[str, Set[str]] = {}
        
        # Pre-computed weighted lists for fast selection
        self._surname_items: List[str] = []
        self._surname_weights: List[int] = []
        self._place_items: List[str] = []
        self._place_weights: List[int] = []
        
        # Combined type lookup: word -> set of WordType values
        self._word_type_map: Dict[str, Set[str]] = {}
    
    def _connect(self) -> None:
        """Establish database connection lazily."""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
    
    def _ensure_initialized(self) -> None:
        """Ensure lookup sets are loaded (idempotent)."""
        if self._initialized:
            return
        self._load_lookup_sets()
        self._initialized = True
    
    def _load_lookup_sets(self) -> None:
        """Load lookup sets and build indices from database."""
        self._connect()
        start_time = time.time()
        print("Loading lookup sets from database...")
        
        self._load_forenames()
        self._load_place_latin_forms()
        self._load_surnames()
        self._load_places()
        self._build_word_type_map()
        
        print(f"Lookup sets loaded in {time.time() - start_time:.3f}s\n")
    
    def _load_forenames(self) -> None:
        """Load forename lookup set and case-based mapping."""
        try:
            cursor = self.conn.execute("""
                SELECT DISTINCT flf.latin_abbreviated
                FROM forename_latin_forms flf
                WHERE flf.latin_abbreviated IS NOT NULL AND flf.latin_abbreviated != ''
            """)
            self._forename_set = {row[0] for row in cursor.fetchall()}
            print(f"  Loaded {len(self._forename_set)} forename lookup entries")
            
            # Build case-based lookup structure
            print("  Building case-based forename lookup structure...")
            load_start = time.time()
            
            cursor = self.conn.execute("""
                SELECT flf.case_name, flf.latin_abbreviated, COALESCE(f.frequency, 0) as frequency
                FROM forename_latin_forms flf
                JOIN forenames f ON flf.forename_id = f.id
                WHERE flf.latin_abbreviated IS NOT NULL AND flf.latin_abbreviated != ''
            """)
            rows = cursor.fetchall()
            
            # Build mappings in single pass
            case_words_dict: Dict[str, Dict[str, int]] = {}
            
            for case_name, latin_abbr, freq in rows:
                if latin_abbr not in self._forename_set:
                    continue
                
                # Track max frequency per word per case
                if case_name not in case_words_dict:
                    case_words_dict[case_name] = {}
                
                current_freq = case_words_dict[case_name].get(latin_abbr, -1)
                if freq > current_freq:
                    case_words_dict[case_name][latin_abbr] = freq
                
                # Build reverse index
                if latin_abbr not in self._word_to_cases:
                    self._word_to_cases[latin_abbr] = set()
                self._word_to_cases[latin_abbr].add(case_name)
            
            # Convert to final format: list of (word, freq) tuples
            for case_name, words_dict in tqdm(case_words_dict.items(), desc="    Converting cases", unit="case"):
                self._forename_cases[case_name] = [(w, max(1, f)) for w, f in words_dict.items()]
            
            print(f"  Built {len(self._forename_cases):,} case groups, {len(self._word_to_cases):,} word mappings in {time.time() - load_start:.3f}s")
            
        except sqlite3.OperationalError as e:
            print(f"  Warning: Could not load forename_latin_forms: {e}")
    
    def _load_place_latin_forms(self) -> None:
        """Load abbreviated place forms mapping."""
        try:
            cursor = self.conn.execute("""
                SELECT plf.latin_abbreviated, p.name
                FROM place_latin_forms plf
                JOIN places p ON plf.place_id = p.id
                WHERE plf.latin_abbreviated IS NOT NULL AND plf.latin_abbreviated != ''
            """)
            for latin_abbr, place_name in cursor.fetchall():
                if latin_abbr not in self._place_latin_forms_map:
                    self._place_latin_forms_map[latin_abbr] = place_name
                    self._place_abbreviated_set.add(latin_abbr)
            
            if self._place_latin_forms_map:
                print(f"  Loaded {len(self._place_latin_forms_map)} place Latin form mappings")
        except sqlite3.OperationalError:
            pass  # Table might not exist
    
    def _load_surnames(self) -> None:
        """Load surname lookup set with frequency weights."""
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
            self._surname_items = [row[0] for row in rows]
            self._surname_weights = [row[1] for row in rows]
            print(f"  Loaded {len(self._surname_set)} surname lookup entries")
        except sqlite3.OperationalError as e:
            print(f"  Warning: Could not load surnames: {e}")
    
    def _load_places(self) -> None:
        """Load place lookup set with frequency weights."""
        try:
            # Try frequency column first, then fallback strategies
            rows = self._try_load_places_with_frequency()
            if rows is None:
                rows = self._try_load_places_with_join()
            if rows is None:
                rows = self._load_places_default()
            
            if rows:
                self._place_set = {row[0] for row in rows}
                self._place_items = [row[0] for row in rows]
                self._place_weights = [max(1, row[1] if len(row) > 1 else 1) for row in rows]
                print(f"  Loaded {len(self._place_set)} place lookup entries")
        except sqlite3.OperationalError as e:
            print(f"  Warning: Could not load places: {e}")
    
    def _try_load_places_with_frequency(self) -> Optional[List]:
        """Try loading places using frequency column."""
        try:
            cursor = self.conn.execute("""
                SELECT p.name, COALESCE(p.frequency, 1) as frequency
                FROM places p
                WHERE p.name IS NOT NULL AND p.name != ''
            """)
            rows = cursor.fetchall()
            return rows if rows else None
        except (sqlite3.OperationalError, sqlite3.DatabaseError):
            return None
    
    def _try_load_places_with_join(self) -> Optional[List]:
        """Try loading places by counting entry_places."""
        try:
            cursor = self.conn.execute("""
                SELECT p.name, COALESCE(COUNT(epl.entry_id), 1) as frequency
                FROM places p
                LEFT JOIN entry_places epl ON p.id = epl.place_id
                WHERE p.name IS NOT NULL AND p.name != ''
                GROUP BY p.id, p.name
            """)
            rows = cursor.fetchall()
            return rows if rows else None
        except sqlite3.OperationalError:
            return None
    
    def _load_places_default(self) -> List:
        """Load places with default frequency of 1."""
        cursor = self.conn.execute("""
            SELECT p.name, 1 as frequency
            FROM places p
            WHERE p.name IS NOT NULL AND p.name != ''
        """)
        return cursor.fetchall()
    
    def _build_word_type_map(self) -> None:
        """Build combined word type lookup map."""
        print("  Building combined word type map...")
        start = time.time()
        
        type_assignments = [
            (self._forename_set, WordType.FORENAME),
            (self._surname_set, WordType.SURNAME),
            (self._place_set, WordType.PLACE),
            (self._place_abbreviated_set, WordType.PLACE_ABBR),
        ]
        
        for word_set, word_type in type_assignments:
            for word in word_set:
                if word not in self._word_type_map:
                    self._word_type_map[word] = set()
                self._word_type_map[word].add(word_type)
        
        print(f"  Built word type map with {len(self._word_type_map)} entries in {time.time() - start:.3f}s")
    
    def _get_word_types(self, word: str, normalized: str) -> Set[str]:
        """Get all word types for a word (checking both original and normalized forms)."""
        types = set()
        if word in self._word_type_map:
            types.update(self._word_type_map[word])
        if normalized != word and normalized in self._word_type_map:
            types.update(self._word_type_map[normalized])
        return types
    
    def _resolve_place_name(self, word: str, normalized: str, word_types: Set[str]) -> Optional[str]:
        """
        Resolve a word to its canonical place name.
        Returns None if the word is not a place.
        """
        # Check direct place match
        if WordType.PLACE in word_types:
            if word in self._place_set:
                return word
            if normalized in self._place_set:
                return normalized
        
        # Check abbreviated form
        if WordType.PLACE_ABBR in word_types:
            if word in self._place_abbreviated_set:
                return self._place_latin_forms_map.get(word)
            if normalized in self._place_abbreviated_set:
                return self._place_latin_forms_map.get(normalized)
        
        return None
    
    def _get_forename_replacement(self, word: str) -> Optional[str]:
        """Get a weighted random replacement forename from the same grammatical case."""
        if word not in self._word_to_cases:
            return None
        
        case_names = self._word_to_cases[word]
        if not case_names:
            return None
        
        # Pick a random case and get candidates
        case_name = random.choice(tuple(case_names))
        case_words = self._forename_cases.get(case_name)
        
        if not case_words or len(case_words) <= 1:
            return None
        
        # Build filtered list excluding the original word
        items = []
        weights = []
        for w, freq in case_words:
            if w != word:
                items.append(w)
                weights.append(freq)
        
        if not items:
            return None
        
        return random.choices(items, weights=weights, k=1)[0]
    
    def _get_surname_replacement(self, word: str) -> Optional[str]:
        """Get a weighted random replacement surname."""
        return weighted_choice_excluding(self._surname_items, self._surname_weights, exclude=word)
    
    def _get_place_replacement(self, place_name: str) -> Optional[str]:
        """Get a weighted random replacement place."""
        return weighted_choice_excluding(self._place_items, self._place_weights, exclude=place_name)
    
    def _try_replace_as_place(self, word: str, normalized: str, word_types: Set[str]) -> Optional[str]:
        """Attempt to replace word as a place name."""
        place_name = self._resolve_place_name(word, normalized, word_types)
        if place_name:
            return self._get_place_replacement(place_name)
        return None
    
    def replace_words(self, text: str) -> Tuple[str, Dict[str, int]]:
        """
        Replace words in text based on database mappings.
        
        Priority logic:
        - Common place names (London, York, etc.) are always treated as places
        - Otherwise: forenames > surnames > places
        
        Returns:
            Tuple of (replaced_text, replacement_counts_dict)
        """
        self._ensure_initialized()
        
        words = text.split()
        result_parts = []
        counts = {'forenames': 0, 'surnames': 0, 'places': 0}
        
        for word in words:
            normalized = normalize_word(word)
            word_types = self._get_word_types(word, normalized)
            
            replacement = None
            replacement_type = None
            
            # Determine word categories
            is_place = WordType.PLACE in word_types or WordType.PLACE_ABBR in word_types
            is_forename = WordType.FORENAME in word_types
            is_surname = WordType.SURNAME in word_types
            
            # Special case: prioritize common place names
            if is_place and (is_forename or is_surname) and normalized.lower() in self.COMMON_PLACE_NAMES:
                replacement = self._try_replace_as_place(word, normalized, word_types)
                replacement_type = 'places'
            
            # Standard priority: forenames > surnames > places
            if replacement is None and is_forename:
                replacement = self._get_forename_replacement(normalized)
                replacement_type = 'forenames'
            
            if replacement is None and is_surname:
                replacement = self._get_surname_replacement(normalized)
                replacement_type = 'surnames'
            
            if replacement is None and is_place:
                replacement = self._try_replace_as_place(word, normalized, word_types)
                replacement_type = 'places'
            
            # Apply replacement or keep original
            if replacement:
                result_parts.append(replacement)
                counts[replacement_type] += 1
            else:
                result_parts.append(word)
        
        return ' '.join(result_parts), counts
    
    def close(self) -> None:
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
    
    if args.seed is not None:
        random.seed(args.seed)
    
    # Validate paths
    model_path = Path(__file__).parent / args.model
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return 1
    
    db_path = Path(__file__).parent / args.db
    if not db_path.exists():
        print(f"Error: Database file not found: {db_path}")
        return 1
    
    model = load_model(model_path)
    replacer = DatabaseReplacer(db_path)
    
    try:
        print("Generating text...")
        generate_start = time.time()
        
        sentence = None
        last_error = None
        for attempt in range(args.tries):
            try:
                sentence = model.make_sentence()
                if sentence:
                    break
            except Exception as e:
                last_error = e
        
        print(f"Text generation: {time.time() - generate_start:.3f}s\n")
        
        if not sentence:
            error_msg = f" Last error: {last_error}" if last_error else ""
            print(f"Error: Could not generate a sentence after {args.tries} attempts.{error_msg}")
            return 1
        
        print(f"Original text: {sentence}\n")
        
        replaced_text, counts = replacer.replace_words(sentence)
        
        print(f"Replaced text: {replaced_text}\n")
        print("Replacements made:")
        print(f"  Forenames: {counts['forenames']}")
        print(f"  Surnames: {counts['surnames']}")
        print(f"  Places: {counts['places']}")
        print(f"  Total: {sum(counts.values())}")
        
        return 0
    finally:
        replacer.close()


if __name__ == "__main__":
    exit(main() or 0)
