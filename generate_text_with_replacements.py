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
from functools import lru_cache
import bisect

try:
    import markovify
except ImportError:
    print("Error: markovify is not installed. Install it with: pip install markovify")
    raise

# Try to import numpy for faster weighted sampling
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Import tqdm with fallback
try:
    from tqdm import tqdm
except (ImportError, ModuleNotFoundError):
    def tqdm(iterable, *args, **kwargs):
        return iterable

# Import the custom LatinText class
from build_markov_model import LatinText

# Punctuation to strip from word endings
TRAILING_PUNCTUATION = ',.;:!?'


def normalize_word(word: str) -> str:
    """Normalize a word by stripping trailing punctuation."""
    return word.rstrip(TRAILING_PUNCTUATION)


class FastWeightedSampler:
    """
    Pre-computed weighted sampler using cumulative weights and binary search.
    Much faster than calling random.choices() repeatedly.
    """
    __slots__ = ('items', 'cumulative_weights', 'total_weight', '_np_probs', '_use_numpy')
    
    def __init__(self, items: List[str], weights: List[int]):
        self.items = items
        self._use_numpy = HAS_NUMPY and len(items) > 100
        
        if self._use_numpy:
            # NumPy-based sampling (fastest for large arrays)
            weights_arr = np.array(weights, dtype=np.float64)
            self.total_weight = weights_arr.sum()
            self._np_probs = weights_arr / self.total_weight
            self.cumulative_weights = None
        else:
            # Pure Python with cumulative weights and binary search
            self.cumulative_weights = []
            total = 0
            for w in weights:
                total += w
                self.cumulative_weights.append(total)
            self.total_weight = total
            self._np_probs = None
    
    def sample(self, exclude: Optional[str] = None) -> Optional[str]:
        """Sample a random item, optionally excluding one value."""
        if not self.items:
            return None
        
        if exclude is None:
            return self._sample_fast()
        
        # With exclusion: sample until we get something different
        # (Efficient when exclusion is rare relative to total items)
        for _ in range(10):  # Try up to 10 times
            result = self._sample_fast()
            if result != exclude:
                return result
        
        # Fallback: filter and resample (rare case)
        return self._sample_with_filter(exclude)
    
    def _sample_fast(self) -> str:
        """Fast sampling without exclusion."""
        if self._use_numpy:
            return self.items[np.random.choice(len(self.items), p=self._np_probs)]
        else:
            r = random.random() * self.total_weight
            idx = bisect.bisect_left(self.cumulative_weights, r)
            return self.items[min(idx, len(self.items) - 1)]
    
    def _sample_with_filter(self, exclude: str) -> Optional[str]:
        """Fallback sampling with exclusion filter."""
        try:
            exclude_idx = self.items.index(exclude)
        except ValueError:
            return self._sample_fast()
        
        # Build filtered sample space
        if self._use_numpy:
            mask = np.ones(len(self.items), dtype=bool)
            mask[exclude_idx] = False
            filtered_probs = self._np_probs[mask]
            if filtered_probs.sum() == 0:
                return None
            filtered_probs = filtered_probs / filtered_probs.sum()
            filtered_items = [self.items[i] for i in range(len(self.items)) if i != exclude_idx]
            return filtered_items[np.random.choice(len(filtered_items), p=filtered_probs)]
        else:
            filtered_items = self.items[:exclude_idx] + self.items[exclude_idx+1:]
            if not filtered_items:
                return None
            return random.choice(filtered_items)


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
    """Constants for word types."""
    FORENAME = 'forename'
    SURNAME = 'surname'
    PLACE = 'place'
    PLACE_ABBR = 'place_abbr'


class DatabaseReplacer:
    """
    Efficient database-backed word replacer with pre-computed samplers.
    Optimized for batch text generation.
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
        self._word_to_cases: Dict[str, Tuple[str, ...]] = {}  # Pre-converted to tuple
        
        # Pre-computed samplers (FastWeightedSampler instances)
        self._surname_sampler: Optional[FastWeightedSampler] = None
        self._place_sampler: Optional[FastWeightedSampler] = None
        self._forename_case_samplers: Dict[str, FastWeightedSampler] = {}
        
        # Combined type lookup: word -> frozenset of WordType values
        self._word_type_map: Dict[str, FrozenSet[str]] = {}
    
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
        """Load lookup sets and build pre-computed samplers."""
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
        """Load forename lookup set and build case-based samplers."""
        try:
            cursor = self.conn.execute("""
                SELECT DISTINCT flf.latin_abbreviated
                FROM forename_latin_forms flf
                WHERE flf.latin_abbreviated IS NOT NULL AND flf.latin_abbreviated != ''
            """)
            self._forename_set = {row[0] for row in cursor.fetchall()}
            print(f"  Loaded {len(self._forename_set)} forename lookup entries")
            
            print("  Building case-based forename lookup structure...")
            load_start = time.time()
            
            cursor = self.conn.execute("""
                SELECT flf.case_name, flf.latin_abbreviated, COALESCE(f.frequency, 1) as frequency
                FROM forename_latin_forms flf
                JOIN forenames f ON flf.forename_id = f.id
                WHERE flf.latin_abbreviated IS NOT NULL AND flf.latin_abbreviated != ''
            """)
            rows = cursor.fetchall()
            
            # Build mappings in single pass
            case_words_dict: Dict[str, Dict[str, int]] = {}
            word_to_cases_temp: Dict[str, Set[str]] = {}
            
            for case_name, latin_abbr, freq in rows:
                if latin_abbr not in self._forename_set:
                    continue
                
                if case_name not in case_words_dict:
                    case_words_dict[case_name] = {}
                
                current_freq = case_words_dict[case_name].get(latin_abbr, -1)
                if freq > current_freq:
                    case_words_dict[case_name][latin_abbr] = max(1, freq)
                
                if latin_abbr not in word_to_cases_temp:
                    word_to_cases_temp[latin_abbr] = set()
                word_to_cases_temp[latin_abbr].add(case_name)
            
            # Convert to tuples (cached) and build FastWeightedSamplers
            for word, cases in word_to_cases_temp.items():
                self._word_to_cases[word] = tuple(cases)
            
            for case_name, words_dict in tqdm(case_words_dict.items(), desc="    Building samplers", unit="case"):
                items = list(words_dict.keys())
                weights = list(words_dict.values())
                self._forename_case_samplers[case_name] = FastWeightedSampler(items, weights)
            
            print(f"  Built {len(self._forename_case_samplers):,} case samplers, {len(self._word_to_cases):,} word mappings in {time.time() - load_start:.3f}s")
            
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
            pass
    
    def _load_surnames(self) -> None:
        """Load surname lookup set and build sampler."""
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
            
            items = [row[0] for row in rows]
            weights = [max(1, row[1]) for row in rows]
            self._surname_sampler = FastWeightedSampler(items, weights)
            
            print(f"  Loaded {len(self._surname_set)} surname lookup entries")
        except sqlite3.OperationalError as e:
            print(f"  Warning: Could not load surnames: {e}")
    
    def _load_places(self) -> None:
        """Load place lookup set and build sampler."""
        try:
            rows = self._try_load_places_with_frequency()
            if rows is None:
                rows = self._try_load_places_with_join()
            if rows is None:
                rows = self._load_places_default()
            
            if rows:
                self._place_set = {row[0] for row in rows}
                items = [row[0] for row in rows]
                weights = [max(1, row[1] if len(row) > 1 else 1) for row in rows]
                self._place_sampler = FastWeightedSampler(items, weights)
                print(f"  Loaded {len(self._place_set)} place lookup entries")
        except sqlite3.OperationalError as e:
            print(f"  Warning: Could not load places: {e}")
    
    def _try_load_places_with_frequency(self) -> Optional[List]:
        try:
            cursor = self.conn.execute("""
                SELECT p.name, COALESCE(p.frequency, 1) as frequency
                FROM places p WHERE p.name IS NOT NULL AND p.name != ''
            """)
            rows = cursor.fetchall()
            return rows if rows else None
        except (sqlite3.OperationalError, sqlite3.DatabaseError):
            return None
    
    def _try_load_places_with_join(self) -> Optional[List]:
        try:
            cursor = self.conn.execute("""
                SELECT p.name, COALESCE(COUNT(epl.entry_id), 1) as frequency
                FROM places p LEFT JOIN entry_places epl ON p.id = epl.place_id
                WHERE p.name IS NOT NULL AND p.name != ''
                GROUP BY p.id, p.name
            """)
            rows = cursor.fetchall()
            return rows if rows else None
        except sqlite3.OperationalError:
            return None
    
    def _load_places_default(self) -> List:
        cursor = self.conn.execute("""
            SELECT p.name, 1 as frequency FROM places p
            WHERE p.name IS NOT NULL AND p.name != ''
        """)
        return cursor.fetchall()
    
    def _build_word_type_map(self) -> None:
        """Build combined word type lookup map with frozen sets."""
        print("  Building combined word type map...")
        start = time.time()
        
        # Build temporary dict with mutable sets
        temp_map: Dict[str, Set[str]] = {}
        
        type_assignments = [
            (self._forename_set, WordType.FORENAME),
            (self._surname_set, WordType.SURNAME),
            (self._place_set, WordType.PLACE),
            (self._place_abbreviated_set, WordType.PLACE_ABBR),
        ]
        
        for word_set, word_type in type_assignments:
            for word in word_set:
                if word not in temp_map:
                    temp_map[word] = set()
                temp_map[word].add(word_type)
        
        # Convert to frozen sets for faster membership testing
        self._word_type_map = {word: frozenset(types) for word, types in temp_map.items()}
        
        print(f"  Built word type map with {len(self._word_type_map)} entries in {time.time() - start:.3f}s")
    
    def _get_word_types(self, word: str, normalized: str) -> FrozenSet[str]:
        """Get all word types for a word."""
        types = self._word_type_map.get(word, frozenset())
        if normalized != word:
            norm_types = self._word_type_map.get(normalized, frozenset())
            if norm_types:
                types = types | norm_types if types else norm_types
        return types
    
    def _resolve_place_name(self, word: str, normalized: str, word_types: FrozenSet[str]) -> Optional[str]:
        """Resolve a word to its canonical place name."""
        if WordType.PLACE in word_types:
            if word in self._place_set:
                return word
            if normalized in self._place_set:
                return normalized
        
        if WordType.PLACE_ABBR in word_types:
            if word in self._place_abbreviated_set:
                return self._place_latin_forms_map.get(word)
            if normalized in self._place_abbreviated_set:
                return self._place_latin_forms_map.get(normalized)
        
        return None
    
    def _get_forename_replacement(self, word: str) -> Optional[str]:
        """Get a weighted random replacement forename."""
        case_names = self._word_to_cases.get(word)
        if not case_names:
            return None
        
        # Pick a random case (tuple already cached)
        case_name = random.choice(case_names)
        sampler = self._forename_case_samplers.get(case_name)
        
        if sampler is None:
            return None
        
        return sampler.sample(exclude=word)
    
    def _get_surname_replacement(self, word: str) -> Optional[str]:
        """Get a weighted random replacement surname."""
        if self._surname_sampler is None:
            return None
        return self._surname_sampler.sample(exclude=word)
    
    def _get_place_replacement(self, place_name: str) -> Optional[str]:
        """Get a weighted random replacement place."""
        if self._place_sampler is None:
            return None
        return self._place_sampler.sample(exclude=place_name)
    
    def _try_replace_as_place(self, word: str, normalized: str, word_types: FrozenSet[str]) -> Optional[str]:
        """Attempt to replace word as a place name."""
        place_name = self._resolve_place_name(word, normalized, word_types)
        if place_name:
            return self._get_place_replacement(place_name)
        return None
    
    def replace_words(self, text: str) -> Tuple[str, Dict[str, int]]:
        """
        Replace words in text based on database mappings.
        Optimized for speed with pre-computed samplers.
        """
        self._ensure_initialized()
        
        words = text.split()
        result_parts = []
        counts = {'forenames': 0, 'surnames': 0, 'places': 0}
        
        for word in words:
            normalized = normalize_word(word)
            word_types = self._get_word_types(word, normalized)
            
            if not word_types:
                result_parts.append(word)
                continue
            
            replacement = None
            replacement_type = None
            
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
            
            if replacement:
                result_parts.append(replacement)
                counts[replacement_type] += 1
            else:
                result_parts.append(word)
        
        return ' '.join(result_parts), counts
    
    def replace_words_batch(self, texts: List[str]) -> List[Tuple[str, Dict[str, int]]]:
        """
        Replace words in multiple texts. More efficient than calling replace_words() in a loop
        because initialization is guaranteed to happen only once.
        """
        self._ensure_initialized()
        return [self.replace_words(text) for text in texts]
    
    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None


def generate_texts_batch(
    model: LatinText,
    replacer: DatabaseReplacer,
    count: int,
    tries: int = 50,
    min_words: int = 2,
    show_progress: bool = True
) -> List[str]:
    """
    Generate multiple texts efficiently.
    
    Args:
        model: Loaded Markov model
        replacer: Initialized DatabaseReplacer
        count: Number of texts to generate
        tries: Max attempts per text
        min_words: Minimum words per text
        show_progress: Show progress bar
    
    Returns:
        List of generated and replaced texts
    """
    # Ensure replacer is initialized before the loop
    replacer._ensure_initialized()
    
    results = []
    failed = 0
    
    iterator = range(count)
    if show_progress:
        iterator = tqdm(iterator, desc="  Generating texts", unit="text")
    
    for _ in iterator:
        # Generate from Markov model
        sentence = None
        for attempt in range(tries):
            try:
                sentence = model.make_sentence()
                if sentence:
                    break
            except Exception:
                continue
        
        if not sentence:
            failed += 1
            continue
        
        # Replace words
        try:
            replaced_text, _ = replacer.replace_words(sentence)
            if len(replaced_text.split()) >= min_words:
                results.append(replaced_text)
            else:
                failed += 1
        except Exception:
            failed += 1
    
    if show_progress and failed > 0:
        print(f"  ({failed} texts failed to generate)")
    
    return results


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
    parser.add_argument('--count', type=int, default=1,
                        help='Number of texts to generate (default: 1)')
    
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
        if HAS_NUMPY:
            np.random.seed(args.seed)
    
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
        if args.count == 1:
            # Single text generation (original behavior)
            print("Generating text...")
            generate_start = time.time()
            
            sentence = None
            for attempt in range(args.tries):
                try:
                    sentence = model.make_sentence()
                    if sentence:
                        break
                except Exception:
                    pass
            
            print(f"Text generation: {time.time() - generate_start:.3f}s\n")
            
            if not sentence:
                print(f"Error: Could not generate a sentence after {args.tries} attempts.")
                return 1
            
            print(f"Original text: {sentence}\n")
            
            replaced_text, counts = replacer.replace_words(sentence)
            
            print(f"Replaced text: {replaced_text}\n")
            print("Replacements made:")
            print(f"  Forenames: {counts['forenames']}")
            print(f"  Surnames: {counts['surnames']}")
            print(f"  Places: {counts['places']}")
            print(f"  Total: {sum(counts.values())}")
        else:
            # Batch generation
            print(f"Generating {args.count} texts...")
            generate_start = time.time()
            
            texts = generate_texts_batch(model, replacer, args.count, args.tries)
            
            elapsed = time.time() - generate_start
            rate = len(texts) / elapsed if elapsed > 0 else 0
            
            print(f"\nGenerated {len(texts)} texts in {elapsed:.2f}s ({rate:.1f} texts/sec)")
            
            if texts:
                print("\nSample outputs:")
                for i, text in enumerate(texts[:5]):
                    print(f"  {i+1}. {text[:100]}{'...' if len(text) > 100 else ''}")
        
        return 0
    finally:
        replacer.close()


if __name__ == "__main__":
    exit(main() or 0)
